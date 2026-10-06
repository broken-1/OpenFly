"""Model-agnostic PUCT search over OpenFly's discrete pose transitions."""

from dataclasses import dataclass, field
import math
from typing import Callable, Sequence

from qwen_vln import (
    ACTION_NAMES,
    PolicyOutput,
    RolloutEvaluation,
)


@dataclass(frozen=True)
class SearchConfig:
    top_k: int = 3
    depth: int = 2
    simulations: int = 12
    puct_c: float = 1.2


@dataclass
class SearchNode:
    pose: list[float]
    action_prefix: tuple[int, ...]
    prior: float
    parent: "SearchNode | None" = None
    children: dict[int, "SearchNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    observation: object = field(default=None, repr=False)
    policy_output: PolicyOutput | None = field(default=None, repr=False)
    evaluation: RolloutEvaluation | None = None

    @property
    def depth(self):
        return len(self.action_prefix)

    @property
    def mean_value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass(frozen=True)
class SearchResult:
    selected_action: int
    base_action: int
    policy_output: PolicyOutput
    candidates: tuple[dict, ...]
    root_action_stats: tuple[dict, ...]
    simulations: tuple[dict, ...]
    max_depth_reached: int


class MCTSSearch:
    def __init__(self, backend, pose_transition: Callable, config: SearchConfig):
        self.backend = backend
        self.pose_transition = pose_transition
        self.config = config

    def search(
        self,
        instruction: str,
        root_pose: Sequence[float],
        root_policy_output: PolicyOutput,
        real_image_history: Sequence,
        action_history: Sequence[int],
        observe: Callable[[Sequence[float]], object],
    ) -> SearchResult:
        root = SearchNode(
            pose=list(root_pose),
            action_prefix=(),
            prior=1.0,
            observation=real_image_history[-1],
            policy_output=root_policy_output,
        )
        self._add_children(root)
        root_candidates = tuple(
            self._candidate_dict(action, child)
            for action, child in root.children.items()
        )

        simulation_logs = []
        max_depth_reached = 0
        for simulation_index in range(self.config.simulations):
            node = root
            path = [root]

            while node.depth < self.config.depth:
                if not node.children:
                    self._expand(
                        node,
                        instruction,
                        real_image_history,
                        action_history,
                        observe,
                    )
                node = self._select_child(node)
                path.append(node)
                if node.visit_count == 0:
                    break

            self._ensure_observation(node, observe)
            if node.evaluation is None:
                node.evaluation = self.backend.evaluate_rollout(
                    instruction=instruction,
                    real_image_history=real_image_history,
                    leaf_image=node.observation,
                    action_history=action_history,
                    action_prefix=node.action_prefix,
                )

            if node.depth < self.config.depth and not node.children:
                self._expand(
                    node,
                    instruction,
                    real_image_history,
                    action_history,
                    observe,
                )

            for path_node in path:
                path_node.visit_count += 1
                path_node.value_sum += node.evaluation.value

            max_depth_reached = max(max_depth_reached, node.depth)
            simulation_logs.append(
                {
                    "simulation": simulation_index + 1,
                    "action_prefix": list(node.action_prefix),
                    "action_prefix_names": [
                        ACTION_NAMES[action] for action in node.action_prefix
                    ],
                    "depth": node.depth,
                    "leaf_value": node.evaluation.value,
                    "evaluation": node.evaluation.to_dict(),
                }
            )

        root_action_stats = tuple(self._root_action_stats(root))
        selected_action = max(
            root.children,
            key=lambda action: (
                root.children[action].visit_count,
                root.children[action].mean_value,
                root.children[action].prior,
            ),
        )
        return SearchResult(
            selected_action=selected_action,
            base_action=root.policy_output.action,
            policy_output=root.policy_output,
            candidates=root_candidates,
            root_action_stats=root_action_stats,
            simulations=tuple(simulation_logs),
            max_depth_reached=max_depth_reached,
        )

    def _expand(
        self,
        node: SearchNode,
        instruction: str,
        real_image_history: Sequence,
        action_history: Sequence[int],
        observe: Callable,
    ):
        self._ensure_observation(node, observe)
        policy_images = list(real_image_history) + [node.observation]
        combined_actions = list(action_history) + list(node.action_prefix)
        node.policy_output = self.backend.policy(
            instruction,
            policy_images,
            combined_actions,
            allow_stop=False,
        )
        self._add_children(node)

    def _add_children(self, node: SearchNode):
        for candidate in node.policy_output.candidates(self.config.top_k):
            node.children[candidate.action] = SearchNode(
                pose=self.pose_transition(node.pose, candidate.action),
                action_prefix=node.action_prefix + (candidate.action,),
                prior=candidate.prior,
                parent=node,
            )

    @staticmethod
    def _ensure_observation(node: SearchNode, observe: Callable):
        if node.observation is None:
            node.observation = observe(node.pose)

    def _select_child(self, node: SearchNode) -> SearchNode:
        unvisited = [child for child in node.children.values() if child.visit_count == 0]
        if unvisited:
            return max(unvisited, key=lambda child: child.prior)

        def puct(child):
            exploration = (
                self.config.puct_c
                * child.prior
                * math.sqrt(node.visit_count)
                / (1 + child.visit_count)
            )
            return child.mean_value + exploration

        return max(node.children.values(), key=puct)

    @staticmethod
    def _candidate_dict(action: int, child: SearchNode):
        return {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "probability": child.parent.policy_output.probabilities[action],
            "prior": child.prior,
        }

    @staticmethod
    def _root_action_stats(root: SearchNode):
        stats = []
        for action, child in root.children.items():
            stats.append(
                {
                    "action": action,
                    "action_name": ACTION_NAMES[action],
                    "visit_count": child.visit_count,
                    "mean_value": child.mean_value,
                    "prior": child.prior,
                    "evaluation": (
                        child.evaluation.to_dict() if child.evaluation else None
                    ),
                }
            )
        return sorted(
            stats,
            key=lambda item: (
                item["visit_count"],
                item["mean_value"],
                item["prior"],
            ),
            reverse=True,
        )

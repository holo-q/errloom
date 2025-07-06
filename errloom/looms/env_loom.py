from typing import Any, Dict, List

from datasets import concatenate_datasets

from errloom.envs.loom import Loom
from errloom.states import Rollout, Rollouts
from errloom.attractors.attractor import Attractor


class RouterLoomAttractor(Attractor):
    """
    Custom attractor for EnvGroup that routes scoring to appropriate environment attractors.
    """

    def __init__(self, env_map: Dict[str, Loom]):
        super().__init__()
        self.env_map = env_map

        # Collect all unique rule function names across all environments
        all_names_set = set()
        for env in env_map.values():
            all_names_set.update(env.attractor.get_rule_names())
        self.all_rule_names = sorted(list(all_names_set))

        self.logger.info(f"EnvGroupAttractor tracking {len(self.all_rule_names)} unique rule functions")

    def get_rule_names(self) -> List[str]:
        """Return all unique rule function names across all environments."""
        return self.all_rule_names

    async def feel(self, out: Rollout, **kwargs) -> Dict[str, float]:
        """
        Route scoring to the appropriate environment's attractor based on task.

        Returns a dict with all rule function names, using 0.0 for functions
        not applicable to this sample's environment.
        """
        # Initialize results with all rule names set to 0.0
        results = {name: 0.0 for name in self.all_rule_names}
        results['rule'] = 0.0

        # Get the appropriate environment
        env = self.env_map.get(out.task)
        if env is None:
            self.logger.warning(f"No environment found for task '{out.task}'")
            return results

        # Score with the environment's attractor
        env_results = await env.attractor.feel(out)
        print(env_results.keys())
        print(env_results['rule'])

        # Update results with scores
        for rule_name, score in env_results.items():
            if rule_name in results:
                results[rule_name] = score
        # dummy scores for all rule functions not in the environment
        for rule_name in self.all_rule_names:
            if rule_name not in env_results:
                results[rule_name] = 0.0
        return results


class RouterLoom(Loom):
    """
    Loom group that acts as a mixture of multiple looms.

    Routes operations to appropriate sub-looms based on the 'task' column.
    """

    def __init__(self,
                 envs: List[Loom],
                 loom_names: List[str] | None = None,
                 **kwargs):
        """
        Initialize LoomGroup with a list of looms.

        Args:
            envs: List of Loom instances
            loom_names: Optional list of names for each loom.
                      If not provided, uses "loom_0", "loom_1", etc.
            **kwargs: Additional arguments passed to parent Loom
        """
        if not envs:
            raise ValueError("LoomGroup requires at least one loom")

        self.envs = envs
        self.loom_names = loom_names or [f"loom_{i}" for i in range(len(envs))]

        if len(self.loom_names) != len(self.envs):
            raise ValueError("Number of loom_names must match number of envs")

        # Create mapping for quick lookup
        self.env_map = {name: env for name, env in zip(self.loom_names, self.envs)}

        # concatenate datasets with task labels
        datasets = []
        eval_datasets = []
        for env, name in zip(self.envs, self.loom_names):
            def add_task(example):
                example['task'] = name
                return example
            env_dataset = env.get_dataset()
            if env_dataset is not None and 'task' not in env_dataset.column_names:
                env_dataset = env_dataset.map(add_task)
            if env_dataset is not None:
                datasets.append(env_dataset)
            env_eval_dataset = env.get_eval_dataset()
            if env_eval_dataset is not None and 'task' not in env_eval_dataset.column_names:
                env_eval_dataset = env_eval_dataset.map(add_task)
            if env_eval_dataset is not None:
                eval_datasets.append(env_eval_dataset)
        dataset = concatenate_datasets(datasets) if datasets else None
        eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None
        # wrap attractors
        attractor = RouterLoomAttractor(self.env_map)

        # initialize parent Loom
        super().__init__(
            roll_dataset=dataset,
            eval_dataset=eval_dataset,
            attractor=attractor,
            **kwargs
        )
        self.logger.info(f"Initialized LoomGroup with {len(envs)} looms: {self.loom_names}")

    def run(self, state) -> Rollout:
        """
        Route rollout to the appropriate sub-environment based on task.

        The task is determined from (in order of priority):
        1. kwargs['task']
        2. info['task']
        3. First environment name (default)
        :param state:
        """
        # Route to appropriate environment
        task = state.row["task"]
        env = self.env_map[task]

        # Pass through all arguments
        return env.run(state)

    def get_loom_for_task(self, task: str) -> Loom:
        """Get the environment instance for a given task name."""
        return self.env_map.get(task, self.envs[0])

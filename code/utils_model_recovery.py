"""Maximum likelihood fitting of foraging models"""

# %%
from typing import Literal

import numpy as np
# from .base import DynamicForagingAgentMLEBase
#from .learn_functions import learn_choice_kernel, learn_RWlike
#from .params.forager_q_learning_params import generate_pydantic_q_learning_params


from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import ConfigDict, Field, create_model, model_validator



from enum import Enum

"""
Pydantic data model of foraging session data for shared validation.

Maybe this is an overkill...
"""

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, field_validator, model_validator


class PhotostimData(BaseModel):
    """Photostimulation data"""

    trial: List[int]
    power: List[float]
    stim_epoch: Optional[List[str]] = None

    class Config:
        """Allow np.ndarray as input"""

        arbitrary_types_allowed = True


class ForagingSessionData(BaseModel):
    """Shared validation for foraging session data"""

    choice_history: np.ndarray
    reward_history: np.ndarray
    p_reward: Optional[np.ndarray] = None
    random_number: Optional[np.ndarray] = None
    autowater_offered: Optional[np.ndarray] = None
    fitted_data: Optional[np.ndarray] = None
    photostim: Optional[PhotostimData] = None

    class Config:
        """Allow np.ndarray as input"""

        arbitrary_types_allowed = True

    @field_validator(
        "choice_history",
        "reward_history",
        "p_reward",
        "random_number",
        "autowater_offered",
        "fitted_data",
        mode="before",
    )
    @classmethod
    def convert_to_ndarray(cls, v, info):
        """Always convert to numpy array"""
        return (
            np.array(
                v,
                dtype=(
                    "bool"
                    if info.field_name in ["reward_history", "autowater_offered"]  # Turn to bool
                    else None
                ),
            )
            if v is not None
            else None
        )

    @model_validator(mode="after")
    def check_all_fields(cls, values):  # noqa: C901
        """Check consistency of all fields"""

        choice_history = values.choice_history
        reward_history = values.reward_history
        p_reward = values.p_reward
        random_number = values.random_number
        autowater_offered = values.autowater_offered
        fitted_data = values.fitted_data
        photostim = values.photostim

        if not np.all(np.isin(choice_history, [0.0, 1.0]) | np.isnan(choice_history)):
            raise ValueError("choice_history must contain only 0, 1, or np.nan.")

        if choice_history.shape != reward_history.shape:
            raise ValueError("choice_history and reward_history must have the same shape.")

        if p_reward.shape != (2, len(choice_history)):
            raise ValueError("reward_probability must have the shape (2, n_trials)")

        if random_number is not None and random_number.shape != p_reward.shape:
            raise ValueError("random_number must have the same shape as reward_probability.")

        if autowater_offered is not None and autowater_offered.shape != choice_history.shape:
            raise ValueError("autowater_offered must have the same shape as choice_history.")

        if fitted_data is not None and fitted_data.shape[0] != choice_history.shape[0]:
            raise ValueError("fitted_data must have the same length as choice_history.")

        if photostim is not None:
            if len(photostim.trial) != len(photostim.power):
                raise ValueError("photostim.trial must have the same length as photostim.power.")
            if photostim.stim_epoch is not None and len(photostim.stim_epoch) != len(
                photostim.power
            ):
                raise ValueError(
                    "photostim.stim_epoch must have the same length as photostim.power."
                )

        return values


def choose_ps(ps, rng=None):
    """
    "Poisson"-choice process
    """
    rng = rng or np.random.default_rng()

    ps = ps / np.sum(ps)
    return np.max(np.argwhere(np.hstack([-1e-16, np.cumsum(ps)]) < rng.random()))


class ParamsSymbols(str, Enum):
    """Symbols for the parameters.

    The order determined the default order of parameters when output as a string.
    """

    loss_count_threshold_mean = R"$\mu_{LC}$"
    loss_count_threshold_std = R"$\sigma_{LC}$"
    learn_rate = R"$\alpha$"
    learn_rate_rew = R"$\alpha_{rew}$"
    learn_rate_unrew = R"$\alpha_{unr}$"
    forget_rate_unchosen = R"$\delta$"
    choice_kernel_step_size = R"$\alpha_{ck}$"
    choice_kernel_relative_weight = R"$w_{ck}$"
    biasL = R"$b_L$"
    softmax_inverse_temperature = R"$\beta$"
    epsilon = R"$\epsilon$"
    threshold = R"$\rho$"  # Adding the threshold parameter with symbol ρ (rho)


def create_pydantic_models_dynamic(
    params_fields: Dict[str, Any],
    fitting_bounds: Dict[str, Tuple[float, float]],
):
    """Create Pydantic models dynamically based on the input fields and fitting bounds."""
    # -- params_model --
    params_model = create_model(
        "ParamsModel",
        **params_fields,
        __config__=ConfigDict(
            extra="forbid",
            validate_assignment=True,
        ),
    )

    # -- fitting_bounds_model --
    fitting_bounds_fields = {}
    for name, (lower, upper) in fitting_bounds.items():
        fitting_bounds_fields[name] = (
            List[float],
            Field(
                default=[lower, upper],
                min_length=2,
                max_length=2,
                description=f"Fitting bounds for {name}",
            ),
        )

    # Add a validator to check the fitting bounds
    def validate_bounds(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        for name, bounds in values.model_dump().items():
            lower_bound, upper_bound = bounds
            if lower_bound > upper_bound:
                raise ValueError(f"Lower bound for {name} must be <= upper bound")
        return values

    fitting_bounds_model = create_model(
        "FittingBoundsModel",
        **fitting_bounds_fields,
        __validators__={"validate_bounds": model_validator(mode="after")(validate_bounds)},
        __config__=ConfigDict(
            extra="forbid",
            validate_assignment=True,
        ),
    )

    return params_model, fitting_bounds_model


def get_params_options(
    params_model,
    default_range=[-np.inf, np.inf],
    para_range_override={},
) -> dict:
    """Get options for the params fields.

    Useful for the Streamlit app.

    Parameters
    ----------
    params_model : Pydantic model
        The Pydantic model for the parameters.
    default_range : list, optional
        The default range for the parameters, by default [-np.inf, np.inf]
        If the range is not specified in the Pydantic model, this default range will be used.
    para_range_override : dict, optional
        The range override for user-specified parameters, by default {}

    Example
    >>> ParamsModel, FittingBoundsModel = generate_pydantic_q_learning_params(
            number_of_learning_rate=1,
            number_of_forget_rate=1,
            choice_kernel="one_step",
            action_selection="softmax",
        )
    >>> params_options = get_params_options(ParamsModel)
    {'learn_rate': {'para_range': [0.0, 1.0],
                    'para_default': 0.5,
                    'para_symbol': <ParamsSymbols.learn_rate: '$\\alpha$'>,
                    'para_desc': 'Learning rate'}, ...
    }

    """
    # Get the schema
    params_schema = params_model.model_json_schema()["properties"]

    # Extract ge and le constraints
    param_options = {}

    for para_name, para_field in params_schema.items():
        default = para_field.get("default", None)
        para_desc = para_field.get("description", "")

        if para_name in para_range_override:
            para_range = para_range_override[para_name]
        else:  # Get from pydantic schema
            para_range = default_range.copy()  # Default range
            # Override the range if specified
            if "minimum" in para_field:
                para_range[0] = para_field["minimum"]
            if "maximum" in para_field:
                para_range[1] = para_field["maximum"]
            para_range = [type(default)(x) for x in para_range]

        param_options[para_name] = dict(
            para_range=para_range,
            para_default=default,
            para_symbol=ParamsSymbols[para_name],
            para_desc=para_desc,
        )
    return param_options


def learn_RWlike(choice, reward, q_value_tminus1, forget_rates, learn_rates):
    """Learning function for Rescorla-Wagner-like model.

    Parameters
    ----------
    choice : int
        this choice
    reward : float
        this reward
    q_value_tminus1 : np.ndarray
        array of old q values
    forget_rates : list
        forget rates for [unchosen, chosen] sides
    learn_rates : _type_
        learning rates for [rewarded, unrewarded] sides

    Returns
    -------
    np.ndarray
        array of new q values
    """
    # Reward-dependent step size ('Hattori2019')
    learn_rate_rew, learn_rate_unrew = learn_rates[0], learn_rates[1]
    if reward:
        learn_rate = learn_rate_rew
    else:
        learn_rate = learn_rate_unrew

    # Choice-dependent forgetting rate ('Hattori2019')
    # Chosen:   Q(n+1) = (1- forget_rate_chosen) * Q(n) + step_size * (Reward - Q(n))
    q_value_t = np.zeros_like(q_value_tminus1)
    K = q_value_tminus1.shape[0]
    q_value_t[choice] = (1 - forget_rates[1]) * q_value_tminus1[choice] + learn_rate * (
        reward - q_value_tminus1[choice]
    )
    # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
    unchosen_idx = [cc for cc in range(K) if cc != choice]
    q_value_t[unchosen_idx] = (1 - forget_rates[0]) * q_value_tminus1[unchosen_idx]
    return q_value_t


def learn_choice_kernel(choice, choice_kernel_tminus1, choice_kernel_step_size):
    """Learning function for choice kernel.

    Parameters
    ----------
    choice : int
        this choice
    choice_kernel_tminus1 : np.ndarray
        array of old choice kernel values
    choice_kernel_step_size : float
        step size for choice kernel

    Returns
    -------
    np.ndarray
        array of new choice kernel values
    """

    # Choice vector
    choice_vector = np.array([0, 0])
    choice_vector[choice] = 1

    # Update choice kernel (see Model 5 of Wilson and Collins, 2019)
    # Note that if chocie_step_size = 1, degenerates to Bari 2019
    # (choice kernel = the last choice only)
    return choice_kernel_tminus1 + choice_kernel_step_size * (choice_vector - choice_kernel_tminus1)



#from .act_functions import act_epsilon_greedy, act_softmax
from typing import Optional
from aind_behavior_gym.dynamic_foraging.task import L, R
from scipy.stats import norm




def act_softmax(
    q_value_t: np.array,
    softmax_inverse_temperature: float,
    bias_terms: np.array,
    choice_kernel_relative_weight=None,
    choice_kernel=None,
    rng=None,
):
    """Given q values and softmax_inverse_temperature, return the choice and choice probability.

    If chocie_kernel is not None, it will sum it into the softmax function like this

    1. Compute adjusted Q values by adding bias terms and choice kernel

        :math:`Q' = \\beta * (Q + w_{ck} * choice\\_kernel) + bias`

        :math:`\\beta` ~ softmax_inverse_temperature

        :math:`w_{ck}` ~ choice_kernel_relative_weight

    2. Compute choice probabilities by softmax function

        :math:`choice\\_prob = exp(Q'_i) / \\sum_i(exp(Q'_i))`

    Parameters
    ----------
    q_value_t : list or np.array
        array of q values, by default 0
    softmax_inverse_temperature : int, optional
        inverse temperature of softmax function, by default 0
    bias_terms : np.array, optional
        _description_, by default 0
    choice_kernel_relative_weight : _type_, optional
        relative strength of choice kernel relative to Q in decision, by default None.
        If not none, choice kernel will have an inverse temperature of
        softmax_inverse_temperature * choice_kernel_relative_weight
    choice_kernel : _type_, optional
        _description_, by default None
    rng : _type_, optional
        random number generator, by default None

    Returns
    -------
    _type_
        _description_
    """

    # -- Compute adjusted Q value --
    # Note that the bias term is outside the temperature term to make it comparable across
    # different softmax_inverse_temperatures.
    # Also, I switched to inverse_temperature from temperature to make
    # the fittings more numerically stable.
    adjusted_Q = softmax_inverse_temperature * q_value_t + bias_terms
    if choice_kernel is not None:
        adjusted_Q += softmax_inverse_temperature * choice_kernel_relative_weight * choice_kernel

    # -- Compute choice probabilities --
    choice_prob = softmax(adjusted_Q, rng=rng)

    # -- Choose action --
    choice = choose_ps(choice_prob, rng=rng)
    return choice, choice_prob


def act_epsilon_greedy(
    q_value_t: np.array,
    epsilon: float,
    bias_terms: np.array,
    choice_kernel=None,
    choice_kernel_relative_weight=None,
    rng=None,
):
    """Action selection by epsilon-greedy method.

    Steps:
    1. Compute adjusted Q values by adding bias terms and choice kernel
        Q' = Q + bias + choice_kernel_relative_weight * choice_kernel
    2. The espilon-greedy method is quivalent to choice probabilities:
        If Q'_L != Q'_R (for simplicity, we assume only two choices)
            choice_prob [(argmax(Q')] = 1 - epsilon / 2
            choice_prob [(argmin(Q'))] = epsilon / 2
        else
            choice_prob [:] = 0.5

    Parameters
    ----------
    q_value_t : np.array
        Current Q-values
    epsilon : float
        Probability of exploration
    bias_terms : np.array
        Bias terms
    choice_kernel : None or np.array, optional
        If not None, it will be added to Q-values, by default None
    choice_kernel_relative_weight : _type_, optional
        If not None, it controls the relative weight of choice kernel, by default None
    rng : _type_, optional
        _description_, by default None
    """
    rng = rng or np.random.default_rng()

    # -- Compute adjusted Q value --
    adjusted_Q = q_value_t + bias_terms
    if choice_kernel is not None:
        adjusted_Q += choice_kernel_relative_weight * choice_kernel

    # -- Compute choice probabilities --
    if adjusted_Q[0] == adjusted_Q[1]:
        choice_prob = np.array([0.5, 0.5])
    else:
        argmax_Q = np.argmax(adjusted_Q)
        choice_prob = np.array([epsilon / 2, epsilon / 2])
        choice_prob[argmax_Q] = 1 - epsilon / 2

    # -- Choose action --
    choice = choose_ps(choice_prob, rng=rng)
    return choice, choice_prob



"""Base class for DynamicForagingAgent with MLE fitting"""

import logging
from typing import Optional, Tuple, Type

import numpy as np
import scipy.optimize as optimize
from aind_behavior_gym.dynamic_foraging.agent import DynamicForagingAgentBase
from aind_behavior_gym.dynamic_foraging.task import DynamicForagingTaskBase
#from aind_dynamic_foraging_basic_analysis import plot_foraging_session
from pydantic import BaseModel

#from .params import ParamsSymbols

from enum import Enum


class ParamsSymbols(str, Enum):
    """Symbols for the parameters.

    The order determined the default order of parameters when output as a string.
    """

    loss_count_threshold_mean = R"$\mu_{LC}$"
    loss_count_threshold_std = R"$\sigma_{LC}$"
    learn_rate = R"$\alpha$"
    learn_rate_rew = R"$\alpha_{rew}$"
    learn_rate_unrew = R"$\alpha_{unr}$"
    forget_rate_unchosen = R"$\delta$"
    choice_kernel_step_size = R"$\alpha_{ck}$"
    choice_kernel_relative_weight = R"$w_{ck}$"
    biasL = R"$b_L$"
    softmax_inverse_temperature = R"$\beta$"
    epsilon = R"$\epsilon$"
    threshold = R"$\rho$"  # Adding the threshold parameter with symbol ρ (rho)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


class DynamicForagingAgentMLEBase(DynamicForagingAgentBase):
    """Base class of "DynamicForagingAgentBase" + "MLE fitting" """

    def __init__(
        self,
        agent_kwargs: dict = {},
        params: dict = {},
        **kwargs,
    ):
        """Init

        Parameters
        ----------
        agent_kwargs : dict, optional
            The hyperparameters that define the agent type, by default {}
            For example, number_of_learning_rate, number_of_forget_rate, etc.
        params : dict, optional
            The kwargs that define the agent's parameters, by default {}
        **kwargs : dict
            Other kwargs that are passed to the base class, like rng's seed, by default {}
        """
        super().__init__(**kwargs)  # Set self.rng etc.

        # Get pydantic model for the parameters and bounds
        self.ParamModel, self.ParamFitBoundModel = self._get_params_model(agent_kwargs)

        # Set and validate the model parameters. Use default parameters if some are not provided
        self.params = self.ParamModel(**params)
        self._get_params_list()  # Get the number of free parameters of the agent etc.

        # Add model fitting related attributes
        self.fitting_result = None
        self.fitting_result_cross_validation = None

        # Some initializations
        self.n_actions = 2
        self.task = None
        self.n_trials = 0  # Should be set in perform or perform_closed_loop

    def _get_params_model(self, agent_kwargs, params) -> Tuple[Type[BaseModel], Type[BaseModel]]:
        """Dynamically generate the Pydantic model for parameters and fitting bounds.

        This should be overridden by the subclass!!
        It should return ParamModel and ParamFitBoundModel here.
        """
        raise NotImplementedError("This should be overridden by the subclass!!")

    def _get_params_list(self):
        """Get the number of free parameters of the agent etc."""
        self.params_list_all = list(self.ParamModel.model_fields.keys())
        self.params_list_frozen = {
            name: field.default
            for name, field in self.ParamModel.model_fields.items()
            if field.frozen
        }  # Parameters that are frozen by construction
        self.params_list_free = list(set(self.params_list_all) - set(self.params_list_frozen))

    def set_params(self, **params):
        """Update the model parameters and validate"""
        # This is safer than model_copy(update) because it will NOT validate the input params
        _params = self.params.model_dump()
        _params.update(params)
        self.params = self.ParamModel(**_params)
        return self.get_params()

    def get_agent_alias(self):
        """Get the agent alias for the model

        Should be overridden by the subclass.
        """
        return ""

    def get_params(self):
        """Get the model parameters in a dictionary format"""
        return self.params.model_dump()

    def get_params_str(self, if_latex=True, if_value=True, decimal=3):
        """Get string of the model parameters

        Parameters
        -----------
        if_latex: bool, optional
            if True, return the latex format of the parameters, by default True
        if_value: bool, optional
            if True, return the value of the parameters, by default True
        decimal: int, optional

        """
        # Sort the parameters by the order of ParamsSymbols
        params_default_order = list(ParamsSymbols.__members__.keys())
        params_list = sorted(
            self.get_params().items(), key=lambda x: params_default_order.index(x[0])
        )

        # Get fixed parameters if any
        if self.fitting_result is not None:
            # Effective fixed parameters (agent's frozen parameters + user specified clamp_params)
            fixed_params = self.fitting_result.fit_settings["clamp_params"].keys()
        else:
            # Frozen parameters (by construction)
            fixed_params = self.params_list_frozen.keys()

        ps = []
        for p in params_list:
            name_str = ParamsSymbols[p[0]] if if_latex else p[0]
            value_str = f" = {p[1]:.{decimal}f}" if if_value else ""
            fix_str = " (fixed)" if p[0] in fixed_params else ""
            ps.append(f"{name_str}{value_str}{fix_str}")

        return ", ".join(ps)

    def get_choice_history(self):
        """Return the history of actions in format that is compatible with other library such as
        aind_dynamic_foraging_basic_analysis
        """
        if self.task is None:
            return None
        # Make sure agent's history is consistent with the task's history and return
        np.testing.assert_array_equal(self.choice_history, self.task.get_choice_history())
        return self.task.get_choice_history()

    def get_reward_history(self):
        """Return the history of reward in format that is compatible with other library such as
        aind_dynamic_foraging_basic_analysis
        """
        if self.task is None:
            return None
        # Make sure agent's history is consistent with the task's history and return
        np.testing.assert_array_equal(self.reward_history, self.task.get_reward_history())
        return self.task.get_reward_history()

    def get_p_reward(self):
        """Return the reward probabilities for each arm in each trial which is compatible with
        other library such as aind_dynamic_foraging_basic_analysis
        """
        if self.task is None:
            return None
        return self.task.get_p_reward()

    def _reset(self):
        """Reset the agent"""
        self.trial = 0

        # MLE agent must have choice_prob
        self.choice_prob = np.full([self.n_actions, self.n_trials], np.nan)
        self.choice_prob[:, 0] = 1 / self.n_actions  # To be strict (actually no use)

        # Choice and reward history have n_trials length
        self.choice_history = np.full(self.n_trials, fill_value=-1, dtype=int)  # Choice history
        # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_history = np.zeros(self.n_trials)

    def perform(
        self,
        task: DynamicForagingTaskBase,
    ):
        """Generative simulation of a task, or "open-loop" simulation

        Override the base class method to include choice_prob caching etc.

        In each trial loop (note when time ticks):
                              agent.act()     task.step()    agent.learn()
            latent variable [t]  -->  choice [t]  --> reward [t] ---->  update latent variable [t+1]
        """
        self.task = task
        self.n_trials = task.num_trials

        # --- Main task loop ---
        self._reset()  # Reset agent
        _, _ = self.task.reset()  # Reset task and get the initial observation
        task_done = False
        while not task_done:
            assert self.trial == self.task.trial  # Ensure the two timers are in sync

            # -- Agent performs an action
            choice, choice_prob = self.act(_)

            # -- Environment steps (enviromnet's timer ticks here!!!)
            _, reward, task_done, _, _ = task.step(choice)

            # -- Update agent history
            self.choice_prob[:, self.trial] = choice_prob
            self.choice_history[self.trial] = choice
            # In Sutton & Barto's convention, reward belongs to the next time step, but we put it
            # in the current time step for the sake of consistency with neuroscience convention
            self.reward_history[self.trial] = reward

            # -- Agent's timer ticks here !!!
            self.trial += 1

            # -- Update q values
            # Note that this will update the q values **after the last trial**, a final update that
            # will not be used to make the next action (because task is *done*) but may be used for
            # correlating with physiology recordings
            self.learn(_, choice, reward, _, task_done)

    def perform_closed_loop(self, fit_choice_history, fit_reward_history):
        """Simulates the agent over a fixed choice and reward history using its params.
        Also called "teacher forcing" or "closed-loop" simulation.

        Unlike .perform() ("generative" simulation), this is called "predictive" simulation,
        which does not need a task and is used for model fitting.
        """
        self.n_trials = len(fit_choice_history)
        self._reset()

        while self.trial <= self.n_trials - 1:
            # -- Compute and cache choice_prob (key to model fitting)
            _, choice_prob = self.act(None)
            self.choice_prob[:, self.trial] = choice_prob

            # -- Clamp history to fit_history
            clamped_choice = fit_choice_history[self.trial].astype(int)
            clamped_reward = fit_reward_history[self.trial]
            self.choice_history[self.trial] = clamped_choice
            self.reward_history[self.trial] = clamped_reward

            # -- Agent's timer ticks here !!!
            self.trial += 1

            # -- Update q values
            self.learn(None, clamped_choice, clamped_reward, None, None)

    def act(self, observation):
        """
        Chooses an action based on the current observation.
        I just copy and paste this from DynamicForagingAgentBase here for clarity.

        Args:
            observation: The current observation from the environment.

        Returns:
            action: The action chosen by the agent.
        """
        raise NotImplementedError("The 'act' method should be overridden by subclasses.")

    def learn(self, observation, action, reward, next_observation, done):
        """
        Updates the agent's knowledge or policy based on the last action and its outcome.
        I just copy and paste this from DynamicForagingAgentBase here for clarity.

        This is the core method that should be implemented by all non-trivial agents.
        It could be Q-learning, policy gradients, neural networks, etc.

        Args:
            observation: The observation before the action was taken.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_observation: The next observation after the action.
            done: Whether the episode has ended.
        """
        raise NotImplementedError("The 'learn' method should be overridden by subclasses.")

    def fit(
        self,
        fit_choice_history,
        fit_reward_history,
        fit_bounds_override: Optional[dict] = {},
        clamp_params: Optional[dict] = {},
        k_fold_cross_validation: Optional[int] = None,
        DE_kwargs: Optional[dict] = {"workers": 1},
    ):
        """Fit the model to the data using differential evolution.

        It handles fit_bounds_override and clamp_params as follows:
        1. It will first clamp the parameters specified in clamp_params
        2. For other parameters, if it is specified in fit_bounds_override, the specified
           bound will be used; otherwise, the bound in the model's ParamFitBounds will be used.

        For example, if params_to_fit and clamp_params are all empty, all parameters will
        be fitted with default bounds in the model's ParamFitBounds.

        Parameters
        ----------
        fit_choice_history : _type_
            _description_
        fit_reward_history : _type_
            _description_
        fit_bounds_override : dict, optional
            Override the default bounds for fitting parameters in ParamFitBounds, by default {}
        clamp_params : dict, optional
            Specify parameters to fix to certain values, by default {}
        k_fold_cross_validation : Optional[int], optional
            Whether to do cross-validation, by default None (no cross-validation).
            If k_fold_cross_validation > 1, it will do k-fold cross-validation and return the
            prediction accuracy of the test set for model comparison.
        DE_kwargs : dict, optional
            kwargs for differential_evolution, by default {'workers': 1}
            For example:
                workers : int
                    Number of workers for differential evolution, by default 1.
                    In CO, fitting a typical session of 1000 trials takes:
                        1 worker: ~100 s
                        4 workers: ~35 s
                        8 workers: ~22 s
                        16 workers: ~20 s
                    (see https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/22075b85360c0a5db475a90bcb025deaa4318f05/notebook/demo_rl_mle_fitting_new_test_time.ipynb) # noqa E501
                    That is to say, the parallel speedup in DE is sublinear. Therefore, given a constant
                    number of total CPUs, it is more efficient to parallelize on the level of session,
                    instead of on DE's workers.

        Returns
        -------
        _type_
            _description_
        """
        # ===== Preparation =====
        # -- Sanity checks --
        # Ensure params_to_fit and clamp_params are not overlapping
        assert set(fit_bounds_override.keys()).isdisjoint(clamp_params.keys())
        # Validate clamp_params
        assert self.ParamModel(**clamp_params)

        # -- Get fit_names and fit_bounds --
        # Validate fit_bounds_override and fill in the missing bounds with default bounds
        fit_bounds = self.ParamFitBoundModel(**fit_bounds_override).model_dump()
        # Add agent's frozen parameters (by construction) to clamp_params (user specified)
        clamp_params = clamp_params.copy()  # Make a copy to avoid modifying the default dict!!
        clamp_params.update(self.params_list_frozen)
        # Remove clamped parameters from fit_bounds
        for name in clamp_params.keys():
            fit_bounds.pop(name)
        # In the remaining parameters, check whether there are still collapsed bounds
        # If yes, clamp them to the collapsed value and remove them from fit_bounds
        _to_remove = []
        for name, bounds in fit_bounds.items():
            if bounds[0] == bounds[1]:
                clamp_params.update({name: bounds[0]})
                _to_remove.append(name)
                logger.warning(
                    f"Parameter {name} is clamped to {bounds[0]} "
                    f"because of collapsed bounds. "
                    f"Please specify it in clamp_params instead."
                )
        for name in _to_remove:
            fit_bounds.pop(name)

        # Get the names of the parameters to fit
        fit_names = list(fit_bounds.keys())
        # Parse bounds
        lower_bounds = [fit_bounds[name][0] for name in fit_names]
        upper_bounds = [fit_bounds[name][1] for name in fit_names]
        # Validate bounds themselves are valid parameters
        try:
            self.ParamModel(**dict(zip(fit_names, lower_bounds)))
            self.ParamModel(**dict(zip(fit_names, upper_bounds)))
        except ValueError as e:
            raise ValueError(
                f"Invalid bounds for {e}.\n"
                f"Bounds must be within the [ge, le] of the ParamModel.\n"
                f"Please check the bounds in fit_bounds_override."
            )

        # # ===== Fit using the whole dataset ======
        logger.info("Fitting the model using the whole dataset...")
        fitting_result = self._optimize_DE(
            fit_choice_history=fit_choice_history,
            fit_reward_history=fit_reward_history,
            fit_names=fit_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            clamp_params=clamp_params,
            fit_trial_set=None,  # None means use all trials to fit
            agent_kwargs=self.agent_kwargs,  # the class AND agent_kwargs fully define the agent
            DE_kwargs=DE_kwargs,
        )

        # -- Save fitting results --
        self.fitting_result = fitting_result

        # -- Rerun the predictive simulation with the fitted params--
        # To fill in the latent variables like q_value and choice_prob
        self.set_params(**fitting_result.params)
        self.perform_closed_loop(fit_choice_history, fit_reward_history)
        # Compute prediction accuracy
        predictive_choice = np.argmax(self.choice_prob, axis=0)
        fitting_result.prediction_accuracy = (
            np.sum(predictive_choice == fit_choice_history) / fitting_result.n_trials
        )

        if k_fold_cross_validation is None:  # Skip cross-validation
            return fitting_result, None

        # ======  Cross-validation ======
        logger.info(
            f"Cross-validating the model using {k_fold_cross_validation}-fold cross-validation..."
        )
        n_trials = len(fit_choice_history)
        trial_numbers_shuffled = np.arange(n_trials)
        self.rng.shuffle(trial_numbers_shuffled)

        prediction_accuracy_fit = []
        prediction_accuracy_test = []
        prediction_accuracy_test_bias_only = []
        fitting_results_all_folds = []

        for kk in range(k_fold_cross_validation):
            logger.info(f"Cross-validation fold {kk+1}/{k_fold_cross_validation}...")
            # -- Split the data --
            test_idx_begin = int(kk * np.floor(n_trials / k_fold_cross_validation))
            test_idx_end = int(
                n_trials
                if (kk == k_fold_cross_validation - 1)
                else (kk + 1) * np.floor(n_trials / k_fold_cross_validation)
            )
            test_set_this = trial_numbers_shuffled[test_idx_begin:test_idx_end]
            fit_set_this = np.hstack(
                (trial_numbers_shuffled[:test_idx_begin], trial_numbers_shuffled[test_idx_end:])
            )

            # -- Fit data using fit_set_this --
            fitting_result_this_fold = self._optimize_DE(
                agent_kwargs=self.agent_kwargs,  # the class AND agent_kwargs fully define the agent
                fit_choice_history=fit_choice_history,
                fit_reward_history=fit_reward_history,
                fit_names=fit_names,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                clamp_params=clamp_params,
                fit_trial_set=fit_set_this,
                DE_kwargs=DE_kwargs,
            )
            fitting_results_all_folds.append(fitting_result_this_fold)

            # -- Compute the prediction accuracy of testing set --
            # Run PREDICTIVE simulation using temp_agent with the fitted parms of this fold
            tmp_agent = self.__class__(params=fitting_result_this_fold.params, **self.agent_kwargs)
            tmp_agent.perform_closed_loop(fit_choice_history, fit_reward_history)

            # Compute prediction accuracy
            predictive_choice_prob_this_fold = np.argmax(tmp_agent.choice_prob, axis=0)

            correct_predicted = predictive_choice_prob_this_fold == fit_choice_history
            prediction_accuracy_fit.append(
                np.sum(correct_predicted[fit_set_this]) / len(fit_set_this)
            )
            prediction_accuracy_test.append(
                np.sum(correct_predicted[test_set_this]) / len(test_set_this)
            )
            # Also return cross-validated prediction_accuracy_bias_only
            if "biasL" in fitting_result_this_fold.params:
                bias_this = fitting_result_this_fold.params["biasL"]
                prediction_correct_bias_only = (
                    int(bias_this <= 0) == fit_choice_history
                )  # If bias_this < 0, bias predicts all rightward choices
                prediction_accuracy_test_bias_only.append(
                    sum(prediction_correct_bias_only[test_set_this]) / len(test_set_this)
                )

        # --- Save all cross_validation results, including raw fitting result of each fold ---
        fitting_result_cross_validation = dict(
            prediction_accuracy_test=prediction_accuracy_test,
            prediction_accuracy_fit=prediction_accuracy_fit,
            prediction_accuracy_test_bias_only=prediction_accuracy_test_bias_only,
            fitting_results_all_folds=fitting_results_all_folds,
        )
        self.fitting_result_cross_validation = fitting_result_cross_validation
        return fitting_result, fitting_result_cross_validation

    def _optimize_DE(
        self,
        agent_kwargs,
        fit_choice_history,
        fit_reward_history,
        fit_names,
        lower_bounds,
        upper_bounds,
        clamp_params,
        fit_trial_set,
        DE_kwargs,
    ):
        """A wrapper of DE fitting for the model. It returns fitting results."""
        # --- Arguments for differential_evolution ---
        kwargs = dict(
            mutation=(0.5, 1),
            recombination=0.7,
            popsize=16,
            polish=True,
            strategy="best1bin",
            disp=False,
            workers=1,
            updating="immediate",
            callback=None,
        )  # Default DE kwargs
        kwargs.update(DE_kwargs)  # Update user specified kwargs
        # Special treatments
        if kwargs["workers"] > 1:
            kwargs["updating"] = "deferred"
        if "seed" in kwargs and isinstance(kwargs["seed"], (int, float)):
            # Convert seed to a numpy random number generator
            # because there seems to be a bug in DE when using int as a seed (not reproducible)
            kwargs["seed"] = np.random.default_rng(kwargs["seed"])

        # --- Heavy lifting here!! ---
        fitting_result = optimize.differential_evolution(
            func=self.__class__._cost_func_for_DE,
            bounds=optimize.Bounds(lower_bounds, upper_bounds),
            args=(
                agent_kwargs,  # Other kwargs to pass to the model
                fit_choice_history,
                fit_reward_history,
                fit_trial_set,  # subset of trials to fit; if empty, use all trials)
                fit_names,  # Pass names so that negLL_func_for_de knows which parameters to fit
                clamp_params,  # Clamped parameters
            ),
            **kwargs,
        )

        # --- Post-processing ---
        fitting_result.fit_settings = dict(
            fit_choice_history=fit_choice_history,
            fit_reward_history=fit_reward_history,
            fit_names=fit_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            clamp_params=clamp_params,
            agent_kwargs=agent_kwargs,
        )
        # Full parameter set
        params = dict(zip(fit_names, fitting_result.x))
        params.update(clamp_params)
        fitting_result.params = params
        fitting_result.k_model = len(fit_names)  # Number of free parameters of the model
        fitting_result.n_trials = len(fit_choice_history)
        fitting_result.log_likelihood = -fitting_result.fun

        fitting_result.AIC = -2 * fitting_result.log_likelihood + 2 * fitting_result.k_model
        fitting_result.BIC = -2 * fitting_result.log_likelihood + fitting_result.k_model * np.log(
            fitting_result.n_trials
        )

        # Likelihood-Per-Trial. See Wilson 2019 (but their formula was wrong...)
        fitting_result.LPT = np.exp(
            fitting_result.log_likelihood / fitting_result.n_trials
        )  # Raw LPT without penality
        fitting_result.LPT_AIC = np.exp(-fitting_result.AIC / 2 / fitting_result.n_trials)
        fitting_result.LPT_BIC = np.exp(-fitting_result.BIC / 2 / fitting_result.n_trials)

        # Always save the result without polishing, regardless of the polish setting
        # (sometimes polishing will move parameters to boundaries, so I add this for sanity check)
        # - About `polish` in DE:
        #   - If `polish=False`, final `x` will be exactly the one in `population` that has the
        #     lowest `population_energy` (typically the first one).
        #     Its energy will also be the final `-log_likelihood`.
        #   - If `polish=True`, an additional gradient-based optimization will
        #     work on `population[0]`, resulting in the final `x`, and override the likelihood
        #     `population_energy[0]` . But it will not change `population[0]`!
        #   - That is to say, `population[0]` is always the result without `polish`.
        #     And if polished, we should rerun a `_cost_func_for_DE` to retrieve
        #     its likelihood, because it has been overridden by `x`.
        idx_lowest_energy = fitting_result.population_energies.argmin()
        x_without_polishing = fitting_result.population[idx_lowest_energy]

        log_likelihood_without_polishing = -self._cost_func_for_DE(
            x_without_polishing,
            agent_kwargs,  # Other kwargs to pass to the model
            fit_choice_history,
            fit_reward_history,
            fit_trial_set,  # subset of trials to fit; if empty, use all trials)
            fit_names,  # Pass names so that negLL_func_for_de knows which parameters to fit
            clamp_params,
        )
        fitting_result.x_without_polishing = x_without_polishing
        fitting_result.log_likelihood_without_polishing = log_likelihood_without_polishing

        params_without_polishing = dict(zip(fit_names, fitting_result.x_without_polishing))
        params_without_polishing.update(clamp_params)
        fitting_result.params_without_polishing = params_without_polishing
        return fitting_result

    @classmethod
    def _cost_func_for_DE(
        cls,
        current_values,  # the current fitting values of params in fit_names (passed by DE)
        # ---- Below are the arguments passed by args. The order must be the same! ----
        agent_kwargs,
        fit_choice_history,
        fit_reward_history,
        fit_trial_set,
        fit_names,
        clamp_params,
    ):
        """The core function that interacts with optimize.differential_evolution().
        For given params, run simulation using clamped history and
        return negative log likelihood.

        Note that this is a class method.
        """

        # -- Parse params and initialize a new agent --
        params = dict(zip(fit_names, current_values))  # Current fitting values
        params.update(clamp_params)  # Add clamped params
        agent = cls(params=params, **agent_kwargs)

        # -- Run **PREDICTIVE** simulation --
        # (clamp the history and do only one forward step on each trial)
        agent.perform_closed_loop(fit_choice_history, fit_reward_history)
        choice_prob = agent.choice_prob

        return negLL(
            choice_prob, fit_choice_history, fit_reward_history, fit_trial_set
        )  # Return total negative log likelihood of the fit_trial_set

    def plot_session(self, if_plot_latent=True):
        """Plot session after .perform(task)

        Parameters
        ----------
        if_plot_latent : bool, optional
            Whether to plot latent variables, by default True
        """
        fig, axes = plot_foraging_session(
            choice_history=self.task.get_choice_history(),
            reward_history=self.task.get_reward_history(),
            p_reward=self.task.get_p_reward(),
        )

        if if_plot_latent:
            # Plot latent variables
            self.plot_latent_variables(axes[0], if_fitted=False)

            # Plot choice_prob
            if "ForagingCompareThreshold" not in self.get_agent_alias():
                axes[0].plot(
                    np.arange(self.n_trials) + 1,
                    self.choice_prob[1] / self.choice_prob.sum(axis=0),
                    lw=0.5,
                    color="green",
                    label="choice_prob(R/R+L)",
                )

        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=3)

        # 　Add the model parameters
        params_str = self.get_params_str()
        fig.suptitle(params_str, fontsize=10, horizontalalignment="left", x=fig.subplotpars.left)

        return fig, axes

    def plot_fitted_session(self, if_plot_latent=True):
        """Plot session after .fit()

        1. choice and reward history will be the history used for fitting
        2. laten variables q_estimate and choice_prob will be plotted
        3. p_reward will be missing (since it is not used for fitting)

        Parameters
        ----------
        if_plot_latent : bool, optional
            Whether to plot latent variables, by default True
        """
        if self.fitting_result is None:
            print("No fitting result found. Please fit the model first.")
            return

        # -- Retrieve fitting results and perform the predictive simiulation
        self.set_params(**self.fitting_result.params)
        fit_choice_history = self.fitting_result.fit_settings["fit_choice_history"]
        fit_reward_history = self.fitting_result.fit_settings["fit_reward_history"]
        self.perform_closed_loop(fit_choice_history, fit_reward_history)

        # -- Plot the target choice and reward history
        # Note that the p_reward could be agnostic to the model fitting.
        fig, axes = plot_foraging_session(
            choice_history=fit_choice_history,
            reward_history=fit_reward_history,
            p_reward=np.full((2, len(fit_choice_history)), np.nan),  # Dummy p_reward
        )

        # -- Plot fitted latent variables and choice_prob --
        if if_plot_latent:
            # Plot latent variables
            self.plot_latent_variables(axes[0], if_fitted=True)
            # Plot fitted choice_prob
            axes[0].plot(
                np.arange(self.n_trials) + 1,
                self.choice_prob[1] / self.choice_prob.sum(axis=0),
                lw=2,
                color="green",
                ls=":",
                label="fitted_choice_prob(R/R+L)",
            )

        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=4)

        # 　Add the model parameters
        params_str = self.get_params_str()
        fig.suptitle(
            f"fitted: {params_str}", fontsize=10, horizontalalignment="left", x=fig.subplotpars.left
        )

        return fig, axes

    def plot_latent_variables(self, ax, if_fitted=False):
        """Add agent-specific latent variables to the plot

        if_fitted: whether the latent variables are from the fitted model (styling purpose)
        """
        pass

    def get_latent_variables(self):
        """Return the latent variables of the agent

        This is agent-specific and should be implemented by the subclass.
        """
        return None

    @staticmethod
    def _fitting_result_to_dict(fitting_result_object, if_include_choice_reward_history=True):
        """Turn each fitting_result object (all data or cross-validation) into a dict

        if_include_choice_reward_history: whether to include choice and reward history in the dict.
        To save space, we may not want to include them for each fold in cross-validation.
        """

        # -- fit_settings --
        fit_settings = fitting_result_object.fit_settings.copy()
        if if_include_choice_reward_history:
            fit_settings["fit_choice_history"] = fit_settings["fit_choice_history"].tolist()
            fit_settings["fit_reward_history"] = fit_settings["fit_reward_history"].tolist()
        else:
            fit_settings.pop("fit_choice_history")
            fit_settings.pop("fit_reward_history")

        # -- fit_stats --
        fit_stats = {}
        fit_stats_fields = [
            "params",
            "log_likelihood",
            "AIC",
            "BIC",
            "LPT",
            "LPT_AIC",
            "LPT_BIC",
            "k_model",
            "n_trials",
            "nfev",
            "nit",
            "success",
            "population",
            "population_energies",
            "params_without_polishing",
            "log_likelihood_without_polishing",
        ]
        for field in fit_stats_fields:
            value = fitting_result_object[field]

            # If numpy array, convert to list
            if isinstance(value, np.ndarray):
                value = value.tolist()
            fit_stats[field] = value

        return {
            "fit_settings": fit_settings,
            **fit_stats,
        }

    def get_fitting_result_dict(self):
        """Return the fitting result in a json-compatible dict for uploading to docDB etc."""
        if self.fitting_result is None:
            print("No fitting result found. Please fit the model first.")
            return

        # -- result of fitting with all data --
        dict_fit_on_whole_data = self._fitting_result_to_dict(
            self.fitting_result, if_include_choice_reward_history=True
        )
        # Add prediction accuracy because it is treated separately for the whole dataset fitting
        dict_fit_on_whole_data["prediction_accuracy"] = self.fitting_result.prediction_accuracy

        # Add class name and agent alias to fit_settings for convenience
        dict_fit_on_whole_data["fit_settings"]["agent_class_name"] = self.__class__.__name__
        dict_fit_on_whole_data["fit_settings"]["agent_alias"] = self.get_agent_alias()

        # -- latent variables --
        latent_variables = self.get_latent_variables()

        # -- Pack all results --
        fitting_result_dict = {
            **dict_fit_on_whole_data,
            "fitted_latent_variables": latent_variables,
        }

        # -- Add cross validation if available --
        if self.fitting_result_cross_validation is not None:
            # Overall goodness of fit
            cross_validation = {
                "prediction_accuracy_test": self.fitting_result_cross_validation[
                    "prediction_accuracy_test"
                ],
                "prediction_accuracy_fit": self.fitting_result_cross_validation[
                    "prediction_accuracy_fit"
                ],
                "prediction_accuracy_test_bias_only": self.fitting_result_cross_validation[
                    "prediction_accuracy_test_bias_only"
                ],
            }

            # Fitting results of each fold
            fitting_results_each_fold = {}
            for kk, fitting_result_fold in enumerate(
                self.fitting_result_cross_validation["fitting_results_all_folds"]
            ):
                fitting_results_each_fold[f"{kk}"] = self._fitting_result_to_dict(
                    fitting_result_fold, if_include_choice_reward_history=False
                )
            cross_validation["fitting_results_each_fold"] = fitting_results_each_fold
            fitting_result_dict["cross_validation"] = cross_validation

        return fitting_result_dict


# -- Helper function --
def negLL(choice_prob, fit_choice_history, fit_reward_history, fit_trial_set=None):
    """Compute total negLL of the trials in fit_trial_set given the data."""

    # Compute negative likelihood
    likelihood_each_trial = choice_prob[
        fit_choice_history.astype(int), range(len(fit_choice_history))
    ]  # Get the actual likelihood for each trial

    # Deal with numerical precision (in rare cases, likelihood can be < 0 or > 1)
    likelihood_each_trial[(likelihood_each_trial <= 0) & (likelihood_each_trial > -1e-5)] = (
        1e-16  # To avoid infinity, which makes the number of zero likelihoods informative!
    )
    likelihood_each_trial[likelihood_each_trial > 1] = 1

    # Return total likelihoods
    if fit_trial_set is None:  # Use all trials
        return -np.sum(np.log(likelihood_each_trial))
    else:  # Use subset of trials in cross-validation
        return -np.sum(np.log(likelihood_each_trial[fit_trial_set]))


class ForagerQLearning(DynamicForagingAgentMLEBase):
    """The familiy of simple Q-learning models."""

    def __init__(
        self,
        number_of_learning_rate: Literal[1, 2] = 2,
        number_of_forget_rate: Literal[0, 1] = 1,
        choice_kernel: Literal["none", "one_step", "full"] = "none",
        action_selection: Literal["softmax", "epsilon-greedy"] = "softmax",
        params: dict = {},
        **kwargs,
    ):
        """Init

        Parameters
        ----------
        number_of_learning_rate : Literal[1, 2], optional
            Number of learning rates, by default 2
            If 1, only one learn_rate will be included in the model.
            If 2, learn_rate_rew and learn_rate_unrew will be included in the model.
        number_of_forget_rate : Literal[0, 1], optional
            Number of forget_rates, by default 1.
            If 0, forget_rate_unchosen will not be included in the model.
            If 1, forget_rate_unchosen will be included in the model.
        choice_kernel : Literal["none", "one_step", "full"], optional
            Choice kernel type, by default "none"
            If "none", no choice kernel will be included in the model.
            If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the last choice
                affects the choice kernel. (Bari2019)
            If "full", both choice_kernel_step_size and choice_kernel_relative_weight
            will be included during fitting
        action_selection : Literal["softmax", "epsilon-greedy"], optional
            Action selection type, by default "softmax"
        params: dict, optional
            Initial parameters of the model, by default {}.
            See the generated Pydantic model in forager_q_learning_params.py for the full
            list of parameters.
        """
        # -- Pack the agent_kwargs --
        self.agent_kwargs = dict(
            number_of_learning_rate=number_of_learning_rate,
            number_of_forget_rate=number_of_forget_rate,
            choice_kernel=choice_kernel,
            action_selection=action_selection,
        )  # Note that the class and self.agent_kwargs fully define the agent

        # -- Initialize the model parameters --
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

        # -- Some agent-family-specific variables --
        self.fit_choice_kernel = False

    def _get_params_model(self, agent_kwargs):
        """Implement the base class method to dynamically generate Pydantic models
        for parameters and fitting bounds for simple Q learning.
        """
        return generate_pydantic_q_learning_params(**agent_kwargs)

    def get_agent_alias(self):
        """Get the agent alias"""
        _ck = {"none": "", "one_step": "_CK1", "full": "_CKfull"}[
            self.agent_kwargs["choice_kernel"]
        ]
        _as = {"softmax": "_softmax", "epsilon-greedy": "_epsi"}[
            self.agent_kwargs["action_selection"]
        ]
        return (
            "QLearning"
            + f"_L{self.agent_kwargs['number_of_learning_rate']}"
            + f"F{self.agent_kwargs['number_of_forget_rate']}"
            + _ck
            + _as
        )

    def _reset(self):
        """Reset the agent"""
        # --- Call the base class reset ---
        super()._reset()

        # --- Agent family specific variables ---
        # Latent variables have n_trials + 1 length to capture the update
        # after the last trial (HH20210726)
        self.q_value = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.q_value[:, 0] = 0  # Initial Q values as 0

        # Always initialize choice_kernel with nan, even if choice_kernel = "none"
        self.choice_kernel = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.choice_kernel[:, 0] = 0  # Initial choice kernel as 0

    def act(self, _):
        """Action selection"""
        # Handle choice kernel
        if self.agent_kwargs["choice_kernel"] == "none":
            choice_kernel = None
            choice_kernel_relative_weight = None
        else:
            choice_kernel = self.choice_kernel[:, self.trial]
            choice_kernel_relative_weight = self.params.choice_kernel_relative_weight

        # Action selection
        if self.agent_kwargs["action_selection"] == "softmax":
            choice, choice_prob = act_softmax(
                q_value_t=self.q_value[:, self.trial],
                softmax_inverse_temperature=self.params.softmax_inverse_temperature,
                bias_terms=np.array([self.params.biasL, 0]),
                # -- Choice kernel --
                choice_kernel=choice_kernel,
                choice_kernel_relative_weight=choice_kernel_relative_weight,
                rng=self.rng,
            )
        elif self.agent_kwargs["action_selection"] == "epsilon-greedy":
            choice, choice_prob = act_epsilon_greedy(
                q_value_t=self.q_value[:, self.trial],
                epsilon=self.params.epsilon,
                bias_terms=np.array([self.params.biasL, 0]),
                # -- Choice kernel --
                choice_kernel=choice_kernel,
                choice_kernel_relative_weight=choice_kernel_relative_weight,
                rng=self.rng,
            )

        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, done):
        """Update Q values

        Note that self.trial already increased by 1 before learn() in the base class
        """

        # Handle params
        if self.agent_kwargs["number_of_learning_rate"] == 1:
            learn_rates = [self.params.learn_rate] * 2
        else:
            learn_rates = [self.params.learn_rate_rew, self.params.learn_rate_unrew]

        if self.agent_kwargs["number_of_forget_rate"] == 0:
            forget_rates = [0, 0]
        else:
            forget_rates = [self.params.forget_rate_unchosen, 0]

        # Update Q values
        self.q_value[:, self.trial] = learn_RWlike(
            choice=choice,
            reward=reward,
            q_value_tminus1=self.q_value[:, self.trial - 1],
            learn_rates=learn_rates,
            forget_rates=forget_rates,
        )

        # Update choice kernel, if used
        if self.agent_kwargs["choice_kernel"] != "none":
            self.choice_kernel[:, self.trial] = learn_choice_kernel(
                choice=choice,
                choice_kernel_tminus1=self.choice_kernel[:, self.trial - 1],
                choice_kernel_step_size=self.params.choice_kernel_step_size,
            )

    def get_latent_variables(self):
        return {
            "q_value": self.q_value.tolist(),
            "choice_kernel": self.choice_kernel.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }

    def plot_latent_variables(self, ax, if_fitted=False):
        """Plot Q values"""
        if if_fitted:
            style = dict(lw=2, ls=":")
            prefix = "fitted_"
        else:
            style = dict(lw=0.5)
            prefix = ""

        x = np.arange(self.n_trials + 1) + 1  # When plotting, we start from 1
        ax.plot(x, self.q_value[L, :], label=f"{prefix}Q(L)", color="red", **style)
        ax.plot(x, self.q_value[R, :], label=f"{prefix}Q(R)", color="blue", **style)

        # Add choice kernel, if used
        if self.agent_kwargs["choice_kernel"] != "none":
            ax.plot(
                x,
                self.choice_kernel[L, :],
                label=f"{prefix}choice_kernel(L)",
                color="purple",
                **style,
            )
            ax.plot(
                x,
                self.choice_kernel[R, :],
                label=f"{prefix}choice_kernel(R)",
                color="cyan",
                **style,
            )
            
            


class DynamicForagingAgentMLEBase(DynamicForagingAgentBase):
    """Base class of "DynamicForagingAgentBase" + "MLE fitting" """

    def __init__(
        self,
        agent_kwargs: dict = {},
        params: dict = {},
        **kwargs,
    ):
        """Init

        Parameters
        ----------
        agent_kwargs : dict, optional
            The hyperparameters that define the agent type, by default {}
            For example, number_of_learning_rate, number_of_forget_rate, etc.
        params : dict, optional
            The kwargs that define the agent's parameters, by default {}
        **kwargs : dict
            Other kwargs that are passed to the base class, like rng's seed, by default {}
        """
        super().__init__(**kwargs)  # Set self.rng etc.

        # Get pydantic model for the parameters and bounds
        self.ParamModel, self.ParamFitBoundModel = self._get_params_model(agent_kwargs)

        # Set and validate the model parameters. Use default parameters if some are not provided
        self.params = self.ParamModel(**params)
        self._get_params_list()  # Get the number of free parameters of the agent etc.

        # Add model fitting related attributes
        self.fitting_result = None
        self.fitting_result_cross_validation = None

        # Some initializations
        self.n_actions = 2
        self.task = None
        self.n_trials = 0  # Should be set in perform or perform_closed_loop

    def _get_params_model(self, agent_kwargs, params) -> Tuple[Type[BaseModel], Type[BaseModel]]:
        """Dynamically generate the Pydantic model for parameters and fitting bounds.

        This should be overridden by the subclass!!
        It should return ParamModel and ParamFitBoundModel here.
        """
        raise NotImplementedError("This should be overridden by the subclass!!")

    def _get_params_list(self):
        """Get the number of free parameters of the agent etc."""
        self.params_list_all = list(self.ParamModel.model_fields.keys())
        self.params_list_frozen = {
            name: field.default
            for name, field in self.ParamModel.model_fields.items()
            if field.frozen
        }  # Parameters that are frozen by construction
        self.params_list_free = list(set(self.params_list_all) - set(self.params_list_frozen))

    def set_params(self, **params):
        """Update the model parameters and validate"""
        # This is safer than model_copy(update) because it will NOT validate the input params
        _params = self.params.model_dump()
        _params.update(params)
        self.params = self.ParamModel(**_params)
        return self.get_params()

    def get_agent_alias(self):
        """Get the agent alias for the model

        Should be overridden by the subclass.
        """
        return ""

    def get_params(self):
        """Get the model parameters in a dictionary format"""
        return self.params.model_dump()

    def get_params_str(self, if_latex=True, if_value=True, decimal=3):
        """Get string of the model parameters

        Parameters
        -----------
        if_latex: bool, optional
            if True, return the latex format of the parameters, by default True
        if_value: bool, optional
            if True, return the value of the parameters, by default True
        decimal: int, optional

        """
        # Sort the parameters by the order of ParamsSymbols
        params_default_order = list(ParamsSymbols.__members__.keys())
        params_list = sorted(
            self.get_params().items(), key=lambda x: params_default_order.index(x[0])
        )

        # Get fixed parameters if any
        if self.fitting_result is not None:
            # Effective fixed parameters (agent's frozen parameters + user specified clamp_params)
            fixed_params = self.fitting_result.fit_settings["clamp_params"].keys()
        else:
            # Frozen parameters (by construction)
            fixed_params = self.params_list_frozen.keys()

        ps = []
        for p in params_list:
            name_str = ParamsSymbols[p[0]] if if_latex else p[0]
            value_str = f" = {p[1]:.{decimal}f}" if if_value else ""
            fix_str = " (fixed)" if p[0] in fixed_params else ""
            ps.append(f"{name_str}{value_str}{fix_str}")

        return ", ".join(ps)

    def get_choice_history(self):
        """Return the history of actions in format that is compatible with other library such as
        aind_dynamic_foraging_basic_analysis
        """
        if self.task is None:
            return None
        # Make sure agent's history is consistent with the task's history and return
        np.testing.assert_array_equal(self.choice_history, self.task.get_choice_history())
        return self.task.get_choice_history()

    def get_reward_history(self):
        """Return the history of reward in format that is compatible with other library such as
        aind_dynamic_foraging_basic_analysis
        """
        if self.task is None:
            return None
        # Make sure agent's history is consistent with the task's history and return
        np.testing.assert_array_equal(self.reward_history, self.task.get_reward_history())
        return self.task.get_reward_history()

    def get_p_reward(self):
        """Return the reward probabilities for each arm in each trial which is compatible with
        other library such as aind_dynamic_foraging_basic_analysis
        """
        if self.task is None:
            return None
        return self.task.get_p_reward()

    def _reset(self):
        """Reset the agent"""
        self.trial = 0

        # MLE agent must have choice_prob
        self.choice_prob = np.full([self.n_actions, self.n_trials], np.nan)
        self.choice_prob[:, 0] = 1 / self.n_actions  # To be strict (actually no use)

        # Choice and reward history have n_trials length
        self.choice_history = np.full(self.n_trials, fill_value=-1, dtype=int)  # Choice history
        # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_history = np.zeros(self.n_trials)

    def perform(
        self,
        task: DynamicForagingTaskBase,
    ):
        """Generative simulation of a task, or "open-loop" simulation

        Override the base class method to include choice_prob caching etc.

        In each trial loop (note when time ticks):
                              agent.act()     task.step()    agent.learn()
            latent variable [t]  -->  choice [t]  --> reward [t] ---->  update latent variable [t+1]
        """
        self.task = task
        self.n_trials = task.num_trials

        # --- Main task loop ---
        self._reset()  # Reset agent
        _, _ = self.task.reset()  # Reset task and get the initial observation
        task_done = False
        while not task_done:
            assert self.trial == self.task.trial  # Ensure the two timers are in sync

            # -- Agent performs an action
            choice, choice_prob = self.act(_)

            # -- Environment steps (enviromnet's timer ticks here!!!)
            _, reward, task_done, _, _ = task.step(choice)

            # -- Update agent history
            self.choice_prob[:, self.trial] = choice_prob
            self.choice_history[self.trial] = choice
            # In Sutton & Barto's convention, reward belongs to the next time step, but we put it
            # in the current time step for the sake of consistency with neuroscience convention
            self.reward_history[self.trial] = reward

            # -- Agent's timer ticks here !!!
            self.trial += 1

            # -- Update q values
            # Note that this will update the q values **after the last trial**, a final update that
            # will not be used to make the next action (because task is *done*) but may be used for
            # correlating with physiology recordings
            self.learn(_, choice, reward, _, task_done)

    def perform_closed_loop(self, fit_choice_history, fit_reward_history):
        """Simulates the agent over a fixed choice and reward history using its params.
        Also called "teacher forcing" or "closed-loop" simulation.

        Unlike .perform() ("generative" simulation), this is called "predictive" simulation,
        which does not need a task and is used for model fitting.
        """
        self.n_trials = len(fit_choice_history)
        self._reset()

        while self.trial <= self.n_trials - 1:
            # -- Compute and cache choice_prob (key to model fitting)
            _, choice_prob = self.act(None)
            self.choice_prob[:, self.trial] = choice_prob

            # -- Clamp history to fit_history
            clamped_choice = fit_choice_history[self.trial].astype(int)
            clamped_reward = fit_reward_history[self.trial]
            self.choice_history[self.trial] = clamped_choice
            self.reward_history[self.trial] = clamped_reward

            # -- Agent's timer ticks here !!!
            self.trial += 1

            # -- Update q values
            self.learn(None, clamped_choice, clamped_reward, None, None)

    def act(self, observation):
        """
        Chooses an action based on the current observation.
        I just copy and paste this from DynamicForagingAgentBase here for clarity.

        Args:
            observation: The current observation from the environment.

        Returns:
            action: The action chosen by the agent.
        """
        raise NotImplementedError("The 'act' method should be overridden by subclasses.")

    def learn(self, observation, action, reward, next_observation, done):
        """
        Updates the agent's knowledge or policy based on the last action and its outcome.
        I just copy and paste this from DynamicForagingAgentBase here for clarity.

        This is the core method that should be implemented by all non-trivial agents.
        It could be Q-learning, policy gradients, neural networks, etc.

        Args:
            observation: The observation before the action was taken.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_observation: The next observation after the action.
            done: Whether the episode has ended.
        """
        raise NotImplementedError("The 'learn' method should be overridden by subclasses.")

    def fit(
        self,
        fit_choice_history,
        fit_reward_history,
        fit_bounds_override: Optional[dict] = {},
        clamp_params: Optional[dict] = {},
        k_fold_cross_validation: Optional[int] = None,
        DE_kwargs: Optional[dict] = {"workers": 1},
    ):
        """Fit the model to the data using differential evolution.

        It handles fit_bounds_override and clamp_params as follows:
        1. It will first clamp the parameters specified in clamp_params
        2. For other parameters, if it is specified in fit_bounds_override, the specified
           bound will be used; otherwise, the bound in the model's ParamFitBounds will be used.

        For example, if params_to_fit and clamp_params are all empty, all parameters will
        be fitted with default bounds in the model's ParamFitBounds.

        Parameters
        ----------
        fit_choice_history : _type_
            _description_
        fit_reward_history : _type_
            _description_
        fit_bounds_override : dict, optional
            Override the default bounds for fitting parameters in ParamFitBounds, by default {}
        clamp_params : dict, optional
            Specify parameters to fix to certain values, by default {}
        k_fold_cross_validation : Optional[int], optional
            Whether to do cross-validation, by default None (no cross-validation).
            If k_fold_cross_validation > 1, it will do k-fold cross-validation and return the
            prediction accuracy of the test set for model comparison.
        DE_kwargs : dict, optional
            kwargs for differential_evolution, by default {'workers': 1}
            For example:
                workers : int
                    Number of workers for differential evolution, by default 1.
                    In CO, fitting a typical session of 1000 trials takes:
                        1 worker: ~100 s
                        4 workers: ~35 s
                        8 workers: ~22 s
                        16 workers: ~20 s
                    (see https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/22075b85360c0a5db475a90bcb025deaa4318f05/notebook/demo_rl_mle_fitting_new_test_time.ipynb) # noqa E501
                    That is to say, the parallel speedup in DE is sublinear. Therefore, given a constant
                    number of total CPUs, it is more efficient to parallelize on the level of session,
                    instead of on DE's workers.

        Returns
        -------
        _type_
            _description_
        """
        # ===== Preparation =====
        # -- Sanity checks --
        # Ensure params_to_fit and clamp_params are not overlapping
        assert set(fit_bounds_override.keys()).isdisjoint(clamp_params.keys())
        # Validate clamp_params
        assert self.ParamModel(**clamp_params)

        # -- Get fit_names and fit_bounds --
        # Validate fit_bounds_override and fill in the missing bounds with default bounds
        fit_bounds = self.ParamFitBoundModel(**fit_bounds_override).model_dump()
        # Add agent's frozen parameters (by construction) to clamp_params (user specified)
        clamp_params = clamp_params.copy()  # Make a copy to avoid modifying the default dict!!
        clamp_params.update(self.params_list_frozen)
        # Remove clamped parameters from fit_bounds
        for name in clamp_params.keys():
            fit_bounds.pop(name)
        # In the remaining parameters, check whether there are still collapsed bounds
        # If yes, clamp them to the collapsed value and remove them from fit_bounds
        _to_remove = []
        for name, bounds in fit_bounds.items():
            if bounds[0] == bounds[1]:
                clamp_params.update({name: bounds[0]})
                _to_remove.append(name)
                logger.warning(
                    f"Parameter {name} is clamped to {bounds[0]} "
                    f"because of collapsed bounds. "
                    f"Please specify it in clamp_params instead."
                )
        for name in _to_remove:
            fit_bounds.pop(name)

        # Get the names of the parameters to fit
        fit_names = list(fit_bounds.keys())
        # Parse bounds
        lower_bounds = [fit_bounds[name][0] for name in fit_names]
        upper_bounds = [fit_bounds[name][1] for name in fit_names]
        # Validate bounds themselves are valid parameters
        try:
            self.ParamModel(**dict(zip(fit_names, lower_bounds)))
            self.ParamModel(**dict(zip(fit_names, upper_bounds)))
        except ValueError as e:
            raise ValueError(
                f"Invalid bounds for {e}.\n"
                f"Bounds must be within the [ge, le] of the ParamModel.\n"
                f"Please check the bounds in fit_bounds_override."
            )

        # # ===== Fit using the whole dataset ======
        logger.info("Fitting the model using the whole dataset...")
        fitting_result = self._optimize_DE(
            fit_choice_history=fit_choice_history,
            fit_reward_history=fit_reward_history,
            fit_names=fit_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            clamp_params=clamp_params,
            fit_trial_set=None,  # None means use all trials to fit
            agent_kwargs=self.agent_kwargs,  # the class AND agent_kwargs fully define the agent
            DE_kwargs=DE_kwargs,
        )

        # -- Save fitting results --
        self.fitting_result = fitting_result

        # -- Rerun the predictive simulation with the fitted params--
        # To fill in the latent variables like q_value and choice_prob
        self.set_params(**fitting_result.params)
        self.perform_closed_loop(fit_choice_history, fit_reward_history)
        # Compute prediction accuracy
        predictive_choice = np.argmax(self.choice_prob, axis=0)
        fitting_result.prediction_accuracy = (
            np.sum(predictive_choice == fit_choice_history) / fitting_result.n_trials
        )

        if k_fold_cross_validation is None:  # Skip cross-validation
            return fitting_result, None

        # ======  Cross-validation ======
        logger.info(
            f"Cross-validating the model using {k_fold_cross_validation}-fold cross-validation..."
        )
        n_trials = len(fit_choice_history)
        trial_numbers_shuffled = np.arange(n_trials)
        self.rng.shuffle(trial_numbers_shuffled)

        prediction_accuracy_fit = []
        prediction_accuracy_test = []
        prediction_accuracy_test_bias_only = []
        fitting_results_all_folds = []

        for kk in range(k_fold_cross_validation):
            logger.info(f"Cross-validation fold {kk+1}/{k_fold_cross_validation}...")
            # -- Split the data --
            test_idx_begin = int(kk * np.floor(n_trials / k_fold_cross_validation))
            test_idx_end = int(
                n_trials
                if (kk == k_fold_cross_validation - 1)
                else (kk + 1) * np.floor(n_trials / k_fold_cross_validation)
            )
            test_set_this = trial_numbers_shuffled[test_idx_begin:test_idx_end]
            fit_set_this = np.hstack(
                (trial_numbers_shuffled[:test_idx_begin], trial_numbers_shuffled[test_idx_end:])
            )

            # -- Fit data using fit_set_this --
            fitting_result_this_fold = self._optimize_DE(
                agent_kwargs=self.agent_kwargs,  # the class AND agent_kwargs fully define the agent
                fit_choice_history=fit_choice_history,
                fit_reward_history=fit_reward_history,
                fit_names=fit_names,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                clamp_params=clamp_params,
                fit_trial_set=fit_set_this,
                DE_kwargs=DE_kwargs,
            )
            fitting_results_all_folds.append(fitting_result_this_fold)

            # -- Compute the prediction accuracy of testing set --
            # Run PREDICTIVE simulation using temp_agent with the fitted parms of this fold
            tmp_agent = self.__class__(params=fitting_result_this_fold.params, **self.agent_kwargs)
            tmp_agent.perform_closed_loop(fit_choice_history, fit_reward_history)

            # Compute prediction accuracy
            predictive_choice_prob_this_fold = np.argmax(tmp_agent.choice_prob, axis=0)

            correct_predicted = predictive_choice_prob_this_fold == fit_choice_history
            prediction_accuracy_fit.append(
                np.sum(correct_predicted[fit_set_this]) / len(fit_set_this)
            )
            prediction_accuracy_test.append(
                np.sum(correct_predicted[test_set_this]) / len(test_set_this)
            )
            # Also return cross-validated prediction_accuracy_bias_only
            if "biasL" in fitting_result_this_fold.params:
                bias_this = fitting_result_this_fold.params["biasL"]
                prediction_correct_bias_only = (
                    int(bias_this <= 0) == fit_choice_history
                )  # If bias_this < 0, bias predicts all rightward choices
                prediction_accuracy_test_bias_only.append(
                    sum(prediction_correct_bias_only[test_set_this]) / len(test_set_this)
                )

        # --- Save all cross_validation results, including raw fitting result of each fold ---
        fitting_result_cross_validation = dict(
            prediction_accuracy_test=prediction_accuracy_test,
            prediction_accuracy_fit=prediction_accuracy_fit,
            prediction_accuracy_test_bias_only=prediction_accuracy_test_bias_only,
            fitting_results_all_folds=fitting_results_all_folds,
        )
        self.fitting_result_cross_validation = fitting_result_cross_validation
        return fitting_result, fitting_result_cross_validation

    def _optimize_DE(
        self,
        agent_kwargs,
        fit_choice_history,
        fit_reward_history,
        fit_names,
        lower_bounds,
        upper_bounds,
        clamp_params,
        fit_trial_set,
        DE_kwargs,
    ):
        """A wrapper of DE fitting for the model. It returns fitting results."""
        # --- Arguments for differential_evolution ---
        kwargs = dict(
            mutation=(0.5, 1),
            recombination=0.7,
            popsize=16,
            polish=True,
            strategy="best1bin",
            disp=False,
            workers=1,
            updating="immediate",
            callback=None,
        )  # Default DE kwargs
        kwargs.update(DE_kwargs)  # Update user specified kwargs
        # Special treatments
        if kwargs["workers"] > 1:
            kwargs["updating"] = "deferred"
        if "seed" in kwargs and isinstance(kwargs["seed"], (int, float)):
            # Convert seed to a numpy random number generator
            # because there seems to be a bug in DE when using int as a seed (not reproducible)
            kwargs["seed"] = np.random.default_rng(kwargs["seed"])

        # --- Heavy lifting here!! ---
        fitting_result = optimize.differential_evolution(
            func=self.__class__._cost_func_for_DE,
            bounds=optimize.Bounds(lower_bounds, upper_bounds),
            args=(
                agent_kwargs,  # Other kwargs to pass to the model
                fit_choice_history,
                fit_reward_history,
                fit_trial_set,  # subset of trials to fit; if empty, use all trials)
                fit_names,  # Pass names so that negLL_func_for_de knows which parameters to fit
                clamp_params,  # Clamped parameters
            ),
            **kwargs,
        )

        # --- Post-processing ---
        fitting_result.fit_settings = dict(
            fit_choice_history=fit_choice_history,
            fit_reward_history=fit_reward_history,
            fit_names=fit_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            clamp_params=clamp_params,
            agent_kwargs=agent_kwargs,
        )
        # Full parameter set
        params = dict(zip(fit_names, fitting_result.x))
        params.update(clamp_params)
        fitting_result.params = params
        fitting_result.k_model = len(fit_names)  # Number of free parameters of the model
        fitting_result.n_trials = len(fit_choice_history)
        fitting_result.log_likelihood = -fitting_result.fun

        fitting_result.AIC = -2 * fitting_result.log_likelihood + 2 * fitting_result.k_model
        fitting_result.BIC = -2 * fitting_result.log_likelihood + fitting_result.k_model * np.log(
            fitting_result.n_trials
        )

        # Likelihood-Per-Trial. See Wilson 2019 (but their formula was wrong...)
        fitting_result.LPT = np.exp(
            fitting_result.log_likelihood / fitting_result.n_trials
        )  # Raw LPT without penality
        fitting_result.LPT_AIC = np.exp(-fitting_result.AIC / 2 / fitting_result.n_trials)
        fitting_result.LPT_BIC = np.exp(-fitting_result.BIC / 2 / fitting_result.n_trials)

        # Always save the result without polishing, regardless of the polish setting
        # (sometimes polishing will move parameters to boundaries, so I add this for sanity check)
        # - About `polish` in DE:
        #   - If `polish=False`, final `x` will be exactly the one in `population` that has the
        #     lowest `population_energy` (typically the first one).
        #     Its energy will also be the final `-log_likelihood`.
        #   - If `polish=True`, an additional gradient-based optimization will
        #     work on `population[0]`, resulting in the final `x`, and override the likelihood
        #     `population_energy[0]` . But it will not change `population[0]`!
        #   - That is to say, `population[0]` is always the result without `polish`.
        #     And if polished, we should rerun a `_cost_func_for_DE` to retrieve
        #     its likelihood, because it has been overridden by `x`.
        idx_lowest_energy = fitting_result.population_energies.argmin()
        x_without_polishing = fitting_result.population[idx_lowest_energy]

        log_likelihood_without_polishing = -self._cost_func_for_DE(
            x_without_polishing,
            agent_kwargs,  # Other kwargs to pass to the model
            fit_choice_history,
            fit_reward_history,
            fit_trial_set,  # subset of trials to fit; if empty, use all trials)
            fit_names,  # Pass names so that negLL_func_for_de knows which parameters to fit
            clamp_params,
        )
        fitting_result.x_without_polishing = x_without_polishing
        fitting_result.log_likelihood_without_polishing = log_likelihood_without_polishing

        params_without_polishing = dict(zip(fit_names, fitting_result.x_without_polishing))
        params_without_polishing.update(clamp_params)
        fitting_result.params_without_polishing = params_without_polishing
        return fitting_result

    @classmethod
    def _cost_func_for_DE(
        cls,
        current_values,  # the current fitting values of params in fit_names (passed by DE)
        # ---- Below are the arguments passed by args. The order must be the same! ----
        agent_kwargs,
        fit_choice_history,
        fit_reward_history,
        fit_trial_set,
        fit_names,
        clamp_params,
    ):
        """The core function that interacts with optimize.differential_evolution().
        For given params, run simulation using clamped history and
        return negative log likelihood.

        Note that this is a class method.
        """

        # -- Parse params and initialize a new agent --
        params = dict(zip(fit_names, current_values))  # Current fitting values
        params.update(clamp_params)  # Add clamped params
        agent = cls(params=params, **agent_kwargs)

        # -- Run **PREDICTIVE** simulation --
        # (clamp the history and do only one forward step on each trial)
        agent.perform_closed_loop(fit_choice_history, fit_reward_history)
        choice_prob = agent.choice_prob

        return negLL(
            choice_prob, fit_choice_history, fit_reward_history, fit_trial_set
        )  # Return total negative log likelihood of the fit_trial_set

    def plot_session(self, if_plot_latent=True):
        """Plot session after .perform(task)

        Parameters
        ----------
        if_plot_latent : bool, optional
            Whether to plot latent variables, by default True
        """
        fig, axes = plot_foraging_session(
            choice_history=self.task.get_choice_history(),
            reward_history=self.task.get_reward_history(),
            p_reward=self.task.get_p_reward(),
        )

        if if_plot_latent:
            # Plot latent variables
            self.plot_latent_variables(axes[0], if_fitted=False)

            # Plot choice_prob
            if "ForagingCompareThreshold" not in self.get_agent_alias():
                axes[0].plot(
                    np.arange(self.n_trials) + 1,
                    self.choice_prob[1] / self.choice_prob.sum(axis=0),
                    lw=0.5,
                    color="green",
                    label="choice_prob(R/R+L)",
                )

        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=3)

        # 　Add the model parameters
        params_str = self.get_params_str()
        fig.suptitle(params_str, fontsize=10, horizontalalignment="left", x=fig.subplotpars.left)

        return fig, axes

    def plot_fitted_session(self, if_plot_latent=True):
        """Plot session after .fit()

        1. choice and reward history will be the history used for fitting
        2. laten variables q_estimate and choice_prob will be plotted
        3. p_reward will be missing (since it is not used for fitting)

        Parameters
        ----------
        if_plot_latent : bool, optional
            Whether to plot latent variables, by default True
        """
        if self.fitting_result is None:
            print("No fitting result found. Please fit the model first.")
            return

        # -- Retrieve fitting results and perform the predictive simiulation
        self.set_params(**self.fitting_result.params)
        fit_choice_history = self.fitting_result.fit_settings["fit_choice_history"]
        fit_reward_history = self.fitting_result.fit_settings["fit_reward_history"]
        self.perform_closed_loop(fit_choice_history, fit_reward_history)

        # -- Plot the target choice and reward history
        # Note that the p_reward could be agnostic to the model fitting.
        fig, axes = plot_foraging_session(
            choice_history=fit_choice_history,
            reward_history=fit_reward_history,
            p_reward=np.full((2, len(fit_choice_history)), np.nan),  # Dummy p_reward
        )

        # -- Plot fitted latent variables and choice_prob --
        if if_plot_latent:
            # Plot latent variables
            self.plot_latent_variables(axes[0], if_fitted=True)
            # Plot fitted choice_prob
            axes[0].plot(
                np.arange(self.n_trials) + 1,
                self.choice_prob[1] / self.choice_prob.sum(axis=0),
                lw=2,
                color="green",
                ls=":",
                label="fitted_choice_prob(R/R+L)",
            )

        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=4)

        # 　Add the model parameters
        params_str = self.get_params_str()
        fig.suptitle(
            f"fitted: {params_str}", fontsize=10, horizontalalignment="left", x=fig.subplotpars.left
        )

        return fig, axes

    def plot_latent_variables(self, ax, if_fitted=False):
        """Add agent-specific latent variables to the plot

        if_fitted: whether the latent variables are from the fitted model (styling purpose)
        """
        pass

    def get_latent_variables(self):
        """Return the latent variables of the agent

        This is agent-specific and should be implemented by the subclass.
        """
        return None

    @staticmethod
    def _fitting_result_to_dict(fitting_result_object, if_include_choice_reward_history=True):
        """Turn each fitting_result object (all data or cross-validation) into a dict

        if_include_choice_reward_history: whether to include choice and reward history in the dict.
        To save space, we may not want to include them for each fold in cross-validation.
        """

        # -- fit_settings --
        fit_settings = fitting_result_object.fit_settings.copy()
        if if_include_choice_reward_history:
            fit_settings["fit_choice_history"] = fit_settings["fit_choice_history"].tolist()
            fit_settings["fit_reward_history"] = fit_settings["fit_reward_history"].tolist()
        else:
            fit_settings.pop("fit_choice_history")
            fit_settings.pop("fit_reward_history")

        # -- fit_stats --
        fit_stats = {}
        fit_stats_fields = [
            "params",
            "log_likelihood",
            "AIC",
            "BIC",
            "LPT",
            "LPT_AIC",
            "LPT_BIC",
            "k_model",
            "n_trials",
            "nfev",
            "nit",
            "success",
            "population",
            "population_energies",
            "params_without_polishing",
            "log_likelihood_without_polishing",
        ]
        for field in fit_stats_fields:
            value = fitting_result_object[field]

            # If numpy array, convert to list
            if isinstance(value, np.ndarray):
                value = value.tolist()
            fit_stats[field] = value

        return {
            "fit_settings": fit_settings,
            **fit_stats,
        }

    def get_fitting_result_dict(self):
        """Return the fitting result in a json-compatible dict for uploading to docDB etc."""
        if self.fitting_result is None:
            print("No fitting result found. Please fit the model first.")
            return

        # -- result of fitting with all data --
        dict_fit_on_whole_data = self._fitting_result_to_dict(
            self.fitting_result, if_include_choice_reward_history=True
        )
        # Add prediction accuracy because it is treated separately for the whole dataset fitting
        dict_fit_on_whole_data["prediction_accuracy"] = self.fitting_result.prediction_accuracy

        # Add class name and agent alias to fit_settings for convenience
        dict_fit_on_whole_data["fit_settings"]["agent_class_name"] = self.__class__.__name__
        dict_fit_on_whole_data["fit_settings"]["agent_alias"] = self.get_agent_alias()

        # -- latent variables --
        latent_variables = self.get_latent_variables()

        # -- Pack all results --
        fitting_result_dict = {
            **dict_fit_on_whole_data,
            "fitted_latent_variables": latent_variables,
        }

        # -- Add cross validation if available --
        if self.fitting_result_cross_validation is not None:
            # Overall goodness of fit
            cross_validation = {
                "prediction_accuracy_test": self.fitting_result_cross_validation[
                    "prediction_accuracy_test"
                ],
                "prediction_accuracy_fit": self.fitting_result_cross_validation[
                    "prediction_accuracy_fit"
                ],
                "prediction_accuracy_test_bias_only": self.fitting_result_cross_validation[
                    "prediction_accuracy_test_bias_only"
                ],
            }

            # Fitting results of each fold
            fitting_results_each_fold = {}
            for kk, fitting_result_fold in enumerate(
                self.fitting_result_cross_validation["fitting_results_all_folds"]
            ):
                fitting_results_each_fold[f"{kk}"] = self._fitting_result_to_dict(
                    fitting_result_fold, if_include_choice_reward_history=False
                )
            cross_validation["fitting_results_each_fold"] = fitting_results_each_fold
            fitting_result_dict["cross_validation"] = cross_validation

        return fitting_result_dict


# -- Helper function --
def negLL(choice_prob, fit_choice_history, fit_reward_history, fit_trial_set=None):
    """Compute total negLL of the trials in fit_trial_set given the data."""

    # Compute negative likelihood
    likelihood_each_trial = choice_prob[
        fit_choice_history.astype(int), range(len(fit_choice_history))
    ]  # Get the actual likelihood for each trial

    # Deal with numerical precision (in rare cases, likelihood can be < 0 or > 1)
    likelihood_each_trial[(likelihood_each_trial <= 0) & (likelihood_each_trial > -1e-5)] = (
        1e-16  # To avoid infinity, which makes the number of zero likelihoods informative!
    )
    likelihood_each_trial[likelihood_each_trial > 1] = 1

    # Return total likelihoods
    if fit_trial_set is None:  # Use all trials
        return -np.sum(np.log(likelihood_each_trial))
    else:  # Use subset of trials in cross-validation
        return -np.sum(np.log(likelihood_each_trial[fit_trial_set]))
    
    


"""Plot foraging session in a standard format.
This is supposed to be reused in plotting real data or simulation data to ensure
a consistent visual representation.
"""

from typing import List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt


def moving_average(a, n=3):
    """
    Compute moving average of a list or array.
    """
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[(n - 1) :] / n  # noqa: E203


def plot_foraging_session_nwb(nwb, **kwargs):
    """
    Wrapper function that extracts fields
    """

    if not hasattr(nwb, "df_trials"):
        print("You need to compute df_trials: nwb_utils.create_trials_df(nwb)")
        return

    if "side_bias" not in nwb.df_trials:
        fig, axes = plot_foraging_session(
            [np.nan if x == 2 else x for x in nwb.df_trials["animal_response"].values],
            nwb.df_trials["earned_reward"].values,
            [nwb.df_trials["reward_probabilityL"], nwb.df_trials["reward_probabilityR"]],
            **kwargs,
        )
    else:
        if "plot_list" not in kwargs:
            kwargs["plot_list"] = ["choice", "reward_prob", "bias"]
        fig, axes = plot_foraging_session(
            [np.nan if x == 2 else x for x in nwb.df_trials["animal_response"].values],
            nwb.df_trials["earned_reward"].values,
            [nwb.df_trials["reward_probabilityL"], nwb.df_trials["reward_probabilityR"]],
            bias=nwb.df_trials["side_bias"].values,
            bias_lower=[x[0] for x in nwb.df_trials["side_bias_confidence_interval"].values],
            bias_upper=[x[1] for x in nwb.df_trials["side_bias_confidence_interval"].values],
            autowater_offered=nwb.df_trials[["auto_waterL", "auto_waterR"]].any(axis=1),
            **kwargs,
        )

    # Add some text info
    # TODO, waiting for AIND metadata to get integrated before adding this info:
    # {df_session.metadata.rig.iloc[0]}, {df_session.metadata.user_name.iloc[0]}\n'
    # f'FORAGING finished {df_session.session_stats.finished_trials.iloc[0]} '
    # f'ignored {df_session.session_stats.ignored_trials.iloc[0]} + '
    # f'AUTOWATER collected {df_session.session_stats.autowater_collected.iloc[0]} '
    # f'ignored {df_session.session_stats.autowater_ignored.iloc[0]}\n'
    # f'FORAGING finished rate {df_session.session_stats.finished_rate.iloc[0]:.2%}, '
    axes[0].text(
        0,
        1.05,
        f"{nwb.session_id}\n"
        f'Total trials {len(nwb.df_trials)}, ignored {np.sum(nwb.df_trials["animal_response"]==2)},'
        f' left {np.sum(nwb.df_trials["animal_response"] == 0)},'
        f' right {np.sum(nwb.df_trials["animal_response"] == 1)}',
        fontsize=8,
        transform=axes[0].transAxes,
    )


def plot_foraging_session(  # noqa: C901
    choice_history: Union[List, np.ndarray],
    reward_history: Union[List, np.ndarray],
    p_reward: Union[List, np.ndarray],
    autowater_offered: Union[List, np.ndarray] = None,
    fitted_data: Union[List, np.ndarray] = None,
    photostim: dict = None,
    valid_range: List = None,
    smooth_factor: int = 5,
    base_color: str = "y",
    ax: plt.Axes = None,
    vertical: bool = False,
    bias: Union[List, np.ndarray] = None,
    bias_lower: Union[List, np.ndarray] = None,
    bias_upper: Union[List, np.ndarray] = None,
    plot_list: List = ["choice", "finished", "reward_prob"],
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot dynamic foraging session.

    Parameters
    ----------
    choice_history : Union[List, np.ndarray]
        Choice history (0 = left choice, 1 = right choice, np.nan = ignored).
    reward_history : Union[List, np.ndarray]
        Reward history (0 = unrewarded, 1 = rewarded).
    p_reward : Union[List, np.ndarray]
        Reward probability for both sides. The size should be (2, len(choice_history)).
    autowater_offered: Union[List, np.ndarray], optional
        If not None, indicates trials where autowater was offered.
    fitted_data : Union[List, np.ndarray], optional
        If not None, overlay the fitted data (e.g. from RL model) on the plot.
    photostim : Dict, optional
        If not None, indicates photostimulation trials. It should be a dictionary with the keys:
            - trial: list of trial numbers
            - power: list of laser power
            - stim_epoch: optional, list of stimulation epochs from
               {"after iti start", "before go cue", "after go cue", "whole trial"}
    valid_range : List, optional
        If not None, add two vertical lines to indicate the valid range where animal was engaged.
    smooth_factor : int, optional
        Smoothing factor for the choice history, by default 5.
    base_color : str, optional
        Base color for the reward probability, by default "yellow".
    ax : plt.Axes, optional
        If not None, use the provided axis to plot, by default None.
    vertical : bool, optional
        If True, plot the session vertically, by default False.

    Returns
    -------
    Tuple[plt.Figure, List[plt.Axes]]
        fig, [ax_choice_reward, ax_reward_schedule]
    """

    # Formatting and sanity checks
    data = ForagingSessionData(
        choice_history=choice_history,
        reward_history=reward_history,
        p_reward=p_reward,
        autowater_offered=autowater_offered,
        fitted_data=fitted_data,
        photostim=PhotostimData(**photostim) if photostim is not None else None,
    )

    choice_history = data.choice_history
    reward_history = data.reward_history
    p_reward = data.p_reward
    autowater_offered = data.autowater_offered
    fitted_data = data.fitted_data
    photostim = data.photostim

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 3) if not vertical else (3, 12), dpi=200)
        plt.subplots_adjust(left=0.1, right=0.8, bottom=0.05, top=0.8)

    if not vertical:
        gs = ax._subplotspec.subgridspec(2, 1, height_ratios=[1, 0.2], hspace=0.1)
        ax_choice_reward = ax.get_figure().add_subplot(gs[0, 0])
        ax_reward_schedule = ax.get_figure().add_subplot(gs[1, 0], sharex=ax_choice_reward)
    else:
        gs = ax._subplotspec.subgridspec(1, 2, width_ratios=[0.2, 1], wspace=0.1)
        ax_choice_reward = ax.get_figure().add_subplot(gs[0, 1])
        ax_reward_schedule = ax.get_figure().add_subplot(gs[0, 0], sharey=ax_choice_reward)

    # == Fetch data ==
    n_trials = len(choice_history)

    p_reward_fraction = p_reward[1, :] / (np.sum(p_reward, axis=0))

    ignored = np.isnan(choice_history)

    if autowater_offered is None:
        rewarded_excluding_autowater = reward_history
        autowater_collected = np.full_like(choice_history, False, dtype=bool)
        autowater_ignored = np.full_like(choice_history, False, dtype=bool)
        unrewarded_trials = ~reward_history & ~ignored
    else:
        rewarded_excluding_autowater = reward_history & ~autowater_offered
        autowater_collected = autowater_offered & ~ignored
        autowater_ignored = autowater_offered & ignored
        unrewarded_trials = ~reward_history & ~ignored & ~autowater_offered

    # == Choice trace ==
    # Rewarded trials (real foraging, autowater excluded)
    xx = np.nonzero(rewarded_excluding_autowater)[0] + 1
    yy = 0.5 + (choice_history[rewarded_excluding_autowater] - 0.5) * 1.4
    yy_temp = choice_history[rewarded_excluding_autowater]
    yy_right = yy_temp[yy_temp > 0.5] + 0.05
    xx_right = xx[yy_temp > 0.5]
    yy_left = yy_temp[yy_temp < 0.5] - 0.05
    xx_left = xx[yy_temp < 0.5]
    if not vertical:
        ax_choice_reward.vlines(
            xx_right,
            yy_right,
            yy_right + 0.1,
            alpha=1,
            linewidth=1,
            color="black",
            label="Rewarded choices",
        )
        ax_choice_reward.vlines(
            xx_left,
            yy_left - 0.1,
            yy_left,
            alpha=1,
            linewidth=1,
            color="black",
        )
    else:
        ax_choice_reward.plot(
            *(xx, yy) if not vertical else [*(yy, xx)],
            "|" if not vertical else "_",
            color="black",
            markersize=10,
            markeredgewidth=2,
            label="Rewarded choices",
        )

    # Unrewarded trials (real foraging; not ignored or autowater trials)
    xx = np.nonzero(unrewarded_trials)[0] + 1
    yy = 0.5 + (choice_history[unrewarded_trials] - 0.5) * 1.4
    yy_temp = choice_history[unrewarded_trials]
    yy_right = yy_temp[yy_temp > 0.5]
    xx_right = xx[yy_temp > 0.5]
    yy_left = yy_temp[yy_temp < 0.5]
    xx_left = xx[yy_temp < 0.5]
    if not vertical:
        ax_choice_reward.vlines(
            xx_right,
            yy_right + 0.05,
            yy_right + 0.1,
            alpha=1,
            linewidth=1,
            color="gray",
            label="Unrewarded choices",
        )
        ax_choice_reward.vlines(
            xx_left,
            yy_left - 0.1,
            yy_left - 0.05,
            alpha=1,
            linewidth=1,
            color="gray",
        )
    else:
        ax_choice_reward.plot(
            *(xx, yy) if not vertical else [*(yy, xx)],
            "|" if not vertical else "_",
            color="gray",
            markersize=6,
            markeredgewidth=1,
            label="Unrewarded choices",
        )

    # Ignored trials
    xx = np.nonzero(ignored & ~autowater_ignored)[0] + 1
    yy = [1.2] * sum(ignored & ~autowater_ignored)
    ax_choice_reward.plot(
        *(xx, yy) if not vertical else [*(yy, xx)],
        "x",
        color="red",
        markersize=3,
        markeredgewidth=0.5,
        label="Ignored",
    )

    # Autowater history
    if autowater_offered is not None:
        # Autowater offered and collected
        xx = np.nonzero(autowater_collected)[0] + 1
        yy = 0.5 + (choice_history[autowater_collected] - 0.5) * 1.4

        yy_temp = choice_history[autowater_collected]
        yy_right = yy_temp[yy_temp > 0.5] + 0.05
        xx_right = xx[yy_temp > 0.5]
        yy_left = yy_temp[yy_temp < 0.5] - 0.05
        xx_left = xx[yy_temp < 0.5]

        if not vertical:
            ax_choice_reward.vlines(
                xx_right,
                yy_right,
                yy_right + 0.1,
                alpha=1,
                linewidth=1,
                color="royalblue",
                label="Autowater collected",
            )
            ax_choice_reward.vlines(
                xx_left,
                yy_left - 0.1,
                yy_left,
                alpha=1,
                linewidth=1,
                color="royalblue",
            )
        else:
            ax_choice_reward.plot(
                *(xx, yy) if not vertical else [*(yy, xx)],
                "|" if not vertical else "_",
                color="royalblue",
                markersize=10,
                markeredgewidth=2,
                label="Autowater collected",
            )

        # Also highlight the autowater offered but still ignored
        xx = np.nonzero(autowater_ignored)[0] + 1
        yy = [1.2] * sum(autowater_ignored)
        ax_choice_reward.plot(
            *(xx, yy) if not vertical else [*(yy, xx)],
            "x",
            color="royalblue",
            markersize=3,
            markeredgewidth=0.5,
            label="Autowater ignored",
        )

    # Base probability
    xx = np.arange(0, n_trials) + 1
    yy = p_reward_fraction
    if "reward_prob" in plot_list:
        ax_choice_reward.plot(
            *(xx, yy) if not vertical else [*(yy, xx)],
            color=base_color,
            label="Base rew. prob.",
            lw=1.5,
        )

    # Smoothed choice history
    y = moving_average(choice_history, smooth_factor) / (
        moving_average(~np.isnan(choice_history), smooth_factor) + 1e-6
    )
    y[y > 100] = np.nan
    x = np.arange(0, len(y)) + int(smooth_factor / 2) + 1
    if "choice" in plot_list:
        ax_choice_reward.plot(
            *(x, y) if not vertical else [*(y, x)],
            linewidth=1.5,
            color="black",
            label="Choice (smooth = %g)" % smooth_factor,
        )

    # finished ratio
    if np.sum(np.isnan(choice_history)):
        x = np.arange(0, len(y)) + int(smooth_factor / 2) + 1
        y = moving_average(~np.isnan(choice_history), smooth_factor)
        if "finished" in plot_list:
            ax_choice_reward.plot(
                *(x, y) if not vertical else [*(y, x)],
                linewidth=0.8,
                color="m",
                alpha=1,
                label="Finished (smooth = %g)" % smooth_factor,
            )

    # Bias
    if ("bias" in plot_list) and (bias is not None):
        bias = (np.array(bias) + 1) / (2)
        bias_lower = (np.array(bias_lower) + 1) / (2)
        bias_upper = (np.array(bias_upper) + 1) / (2)
        bias_lower[bias_lower < 0] = 0
        bias_upper[bias_upper > 1] = 1
        ax_choice_reward.plot(xx, bias, color="g", lw=1.5, label="bias")
        ax_choice_reward.fill_between(xx, bias_lower, bias_upper, color="g", alpha=0.25)
        ax_choice_reward.plot(xx, [0.5] * len(xx), color="g", linestyle="--", alpha=0.2, lw=1)

    # add valid ranage
    if valid_range is not None:
        add_range = ax_choice_reward.axhline if vertical else ax_choice_reward.axvline
        add_range(valid_range[0], color="m", ls="--", lw=1, label="motivation good")
        add_range(valid_range[1], color="m", ls="--", lw=1)

    # For each session, if any fitted_data
    if fitted_data is not None:
        x = np.arange(0, n_trials)
        y = fitted_data
        ax_choice_reward.plot(*(x, y) if not vertical else [*(y, x)], linewidth=1.5, label="model")

    # == photo stim ==
    if photostim is not None:

        trial = data.photostim.trial
        power = data.photostim.power
        stim_epoch = data.photostim.stim_epoch

        if stim_epoch is not None:
            edgecolors = [PHOTOSTIM_EPOCH_MAPPING[t] for t in stim_epoch]
        else:
            edgecolors = "darkcyan"

        x = trial
        y = np.ones_like(trial) + 0.4
        _ = ax_choice_reward.scatter(
            *(x, y) if not vertical else [*(y, x)],
            s=np.array(power) * 2,
            edgecolors=edgecolors,
            marker="v" if not vertical else "<",
            facecolors="none",
            linewidth=0.5,
            label="photostim",
        )

    # p_reward
    xx = np.arange(0, n_trials) + 1
    ll = p_reward[0, :]
    rr = p_reward[1, :]
    ax_reward_schedule.plot(
        *(xx, rr) if not vertical else [*(rr, xx)], color="b", label="p_right", lw=1
    )
    ax_reward_schedule.plot(
        *(xx, ll) if not vertical else [*(ll, xx)], color="r", label="p_left", lw=1
    )
    ax_reward_schedule.legend(fontsize=5, ncol=1, loc="upper left", bbox_to_anchor=(0, 1))

    if not vertical:
        ax_choice_reward.set_yticks([0, 1, 1.2])
        ax_choice_reward.set_yticklabels(["Left", "Right", "Ignored"])
        ax_choice_reward.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.4, 1.3), ncol=3)

        ax_choice_reward.spines["top"].set_visible(False)
        ax_choice_reward.spines["right"].set_visible(False)
        ax_choice_reward.spines["bottom"].set_visible(False)
        ax_choice_reward.tick_params(labelbottom=False)
        ax_choice_reward.xaxis.set_ticks_position("none")
        ax_choice_reward.set_ylim([-0.15, 1.25])

        ax_reward_schedule.set_ylim([0, 1])
        ax_reward_schedule.spines["top"].set_visible(False)
        ax_reward_schedule.spines["right"].set_visible(False)
        ax_reward_schedule.spines["bottom"].set_bounds(0, n_trials)
        ax_reward_schedule.set(xlabel="Trial number")

    else:
        ax_choice_reward.set_xticks([0, 1])
        ax_choice_reward.set_xticklabels(["Left", "Right"])
        ax_choice_reward.invert_yaxis()
        ax_choice_reward.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0, 1.05), ncol=3)

        # ax_choice_reward.set_yticks([])
        ax_choice_reward.spines["top"].set_visible(False)
        ax_choice_reward.spines["right"].set_visible(False)
        ax_choice_reward.spines["left"].set_visible(False)
        ax_choice_reward.tick_params(labelleft=False)
        ax_choice_reward.yaxis.set_ticks_position("none")

        ax_reward_schedule.set_xlim([0, 1])
        ax_reward_schedule.spines["top"].set_visible(False)
        ax_reward_schedule.spines["right"].set_visible(False)
        ax_reward_schedule.spines["left"].set_bounds(0, n_trials)
        ax_reward_schedule.set(ylabel="Trial number")

    ax.remove()
    plt.tight_layout()

    return ax_choice_reward.get_figure(), [ax_choice_reward, ax_reward_schedule]




def generate_pydantic_q_learning_params(
    number_of_learning_rate: Literal[1, 2] = 2,
    number_of_forget_rate: Literal[0, 1] = 1,
    choice_kernel: Literal["none", "one_step", "full"] = "none",
    action_selection: Literal["softmax", "epsilon-greedy"] = "softmax",
) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Dynamically generate Pydantic models for Q-learning agent parameters.

    All default values are hard-coded in this function. But when instantiating the model,
    you can always override the default values, both the params_fields and the fitting bounds.

    Parameters
    ----------
    number_of_learning_rate : Literal[1, 2], optional
        Number of learning rates, by default 2
        If 1, only one learn_rate will be included in the model.
        If 2, learn_rate_rew and learn_rate_unrew will be included in the model.
    number_of_forget_rate : Literal[0, 1], optional
        Number of forget_rates, by default 1.
        If 0, forget_rate_unchosen will not be included in the model.
        If 1, forget_rate_unchosen will be included in the model.
    choice_kernel : Literal["none", "one_step", "full"], optional
        Choice kernel type, by default "none"
        If "none", no choice kernel will be included in the model.
        If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the previous choice
            affects the choice kernel. (Bari2019)
        If "full", both choice_kernel_step_size and choice_kernel_relative_weight will be included
    action_selection : Literal["softmax", "epsilon-greedy"], optional
        Action selection type, by default "softmax"
    """

    # ====== Define common fields and constraints ======
    params_fields = {}
    fitting_bounds = {}

    # -- Handle learning rate fields --
    _add_learning_rate_fields(params_fields, fitting_bounds, number_of_learning_rate)

    # -- Handle forget rate field --
    _add_forget_rate_fields(params_fields, fitting_bounds, number_of_forget_rate)

    # -- Handle choice kernel fields --
    _add_choice_kernel_fields(params_fields, fitting_bounds, choice_kernel)

    # -- Handle action selection fields --
    _add_action_selection_fields(params_fields, fitting_bounds, action_selection)

    # ====== Dynamically create the pydantic models =====
    return create_pydantic_models_dynamic(params_fields, fitting_bounds)


def _add_learning_rate_fields(params_fields, fitting_bounds, number_of_learning_rate):
    """Add learning rate fields to the params_fields and fitting_bounds dictionaries."""
    assert number_of_learning_rate in [1, 2], "number_of_learning_rate must be 1 or 2"
    if number_of_learning_rate == 1:
        params_fields["learn_rate"] = (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="Learning rate"),
        )
        fitting_bounds["learn_rate"] = (0.0, 1.0)
    elif number_of_learning_rate == 2:
        params_fields["learn_rate_rew"] = (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="Learning rate for rewarded choice"),
        )
        fitting_bounds["learn_rate_rew"] = (0.0, 1.0)
        params_fields["learn_rate_unrew"] = (
            float,
            Field(default=0.1, ge=0.0, le=1.0, description="Learning rate for unrewarded choice"),
        )
        fitting_bounds["learn_rate_unrew"] = (0.0, 1.0)


def _add_forget_rate_fields(params_fields, fitting_bounds, number_of_forget_rate):
    """Add forget rate fields to the params_fields and fitting_bounds dictionaries."""
    assert number_of_forget_rate in [0, 1], "number_of_forget_rate must be 0 or 1"
    if number_of_forget_rate == 1:
        params_fields["forget_rate_unchosen"] = (
            float,
            Field(default=0.2, ge=0.0, le=1.0, description="Forgetting rate for unchosen side"),
        )
        fitting_bounds["forget_rate_unchosen"] = (0.0, 1.0)


def _add_choice_kernel_fields(params_fields, fitting_bounds, choice_kernel):
    """Add choice kernel fields to the params_fields and fitting_bounds dictionaries."""
    assert choice_kernel in [
        "none",
        "one_step",
        "full",
    ], "choice_kernel must be 'none', 'one_step', or 'full'"

    if choice_kernel == "none":
        return

    params_fields["choice_kernel_relative_weight"] = (
        float,
        Field(
            default=0.1,
            ge=0.0,
            description=(
                "Relative weight of choice kernel (very sensitive, should be quite small)"
            ),
        ),
    )
    fitting_bounds["choice_kernel_relative_weight"] = (0.0, 1.0)

    if choice_kernel == "full":
        params_fields["choice_kernel_step_size"] = (
            float,
            Field(
                default=0.1,
                ge=0.0,
                le=1.0,
                description="Step size for choice kernel (1.0 means only previous choice)",
            ),
        )
        fitting_bounds["choice_kernel_step_size"] = (0.0, 1.0)
    elif choice_kernel == "one_step":
        # If choice kernel is one-step (only the previous choice affects the choice kernel like
        # in Bari2019), set choice_kernel_step_size to 1.0
        params_fields["choice_kernel_step_size"] = (
            float,
            Field(
                default=1.0,
                ge=1.0,
                le=1.0,
                description="Step size for choice kernel == 1 (one-step choice kernel)",
                frozen=True,  # To indicate that this field is clamped by construction
            ),
        )
        fitting_bounds["choice_kernel_step_size"] = (1.0, 1.0)


def _add_action_selection_fields(params_fields, fitting_bounds, action_selection):
    """Add action selection fields to the params_fields and fitting_bounds dictionaries."""
    # Always include biasL
    params_fields["biasL"] = (float, Field(default=0.0, description="Left bias for softmax"))
    fitting_bounds["biasL"] = (-5.0, 5.0)

    if action_selection == "softmax":
        params_fields["softmax_inverse_temperature"] = (
            float,
            Field(default=10.0, ge=0.0, description="Softmax inverse temperature"),
        )
        fitting_bounds["softmax_inverse_temperature"] = (0.0, 100.0)
    elif action_selection == "epsilon-greedy":
        params_fields["epsilon"] = (
            float,
            Field(default=0.1, ge=0.0, le=1.0, description="Epsilon for epsilon-greedy"),
        )
        fitting_bounds["epsilon"] = (0.0, 1.0)
    else:
        raise ValueError("action_selection must be 'softmax' or 'epsilon-greedy'")
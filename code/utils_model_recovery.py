"""Maximum likelihood fitting of foraging models"""

# %%
from typing import Literal

import numpy as np
# from .base import DynamicForagingAgentMLEBase
#from .learn_functions import learn_choice_kernel, learn_RWlike
#from .params.forager_q_learning_params import generate_pydantic_q_learning_params


from typing import Any, Dict, List, Tuple


from pydantic import ConfigDict, Field, create_model, model_validator



from enum import Enum

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

import contextlib
class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass



def callback_history(x, **kargs):
    '''
    Store the intermediate DE results. I have to use global variable as a workaround. Any better ideas?
    '''
    global fit_history
    fit_history.append(x)
    
    return
    

def generate_true_paras(para_bounds, n_models = 5, method = 'random_uniform'):
    
    if method == 'linspace':
        p1 = np.linspace(para_bounds[0][0], para_bounds[1][0], n_models[0])
        p2 = np.linspace(para_bounds[0][1], para_bounds[1][1], n_models[1])
        pp1,pp2 = np.meshgrid(p1,p2)
        true_paras = np.vstack([pp1.flatten(), pp2.flatten()])
        
        return true_paras

    elif method == 'random_uniform':
        n_paras = len(para_bounds[0])
        
        true_paras = np.zeros([n_paras, n_models])

        for n in range(n_models):
            true_paras_this = []
            for pp in range(n_paras):
                true_paras_this.append(np.random.uniform(para_bounds[0][pp], para_bounds[1][pp]))
            true_paras[:,n] = true_paras_this
                
        return true_paras

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
#from aind_behavior_gym.dynamic_foraging.task import L, R
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

def fit_each_init(forager, fit_names, fit_bounds, choice_history, reward_history, session_num, fit_method, callback):
    '''
    For local optimizers, fit using ONE certain initial condition    
    '''
    x0 = []
    for lb,ub in zip(fit_bounds[0], fit_bounds[1]):
        x0.append(np.random.uniform(lb,ub))
        
    # Append the initial point
    if callback != None: callback_history(x0)
        
    fitting_result = optimize.minimize(negLL_func, x0, args = (forager, fit_names, choice_history, reward_history, session_num, {}, []), method = fit_method,
                                       bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), callback = callback, )
    return fitting_result

def negLL_func(fit_value, *argss):
    '''
    Compute negative likelihood (Core func)
    '''
    # Arguments interpretation
    forager, fit_names, choice_history, reward_history, session_num, para_fixed, fit_set = argss
    
    kwargs_all = {'forager': forager, **para_fixed}  # **kargs includes all other fixed parameters
    for (nn, vv) in zip(fit_names, fit_value):
        kwargs_all = {**kwargs_all, nn:vv}

    # Put constraint hack here!!
    if 'tau2' in kwargs_all:
        if kwargs_all['tau2'] < kwargs_all['tau1']:
            return np.inf
        
    # Handle data from different sessions
    if session_num is None:
        session_num = np.zeros_like(choice_history)[0]  # Regard as one session
    
    unique_session = np.unique(session_num)
    likelihood_all_trial = []
    
    # -- For each session --
    for ss in unique_session:
        # Data in this session
        choice_this = choice_history[:, session_num == ss]
        reward_this = reward_history[:, session_num == ss]
        
        # Run **PREDICTIVE** simulation    
        bandit = BanditModel(**kwargs_all, fit_choice_history = choice_this, fit_reward_history = reward_this)  # Into the fitting mode
        bandit.simulate()
        
        # Compute negative likelihood
        predictive_choice_prob = bandit.predictive_choice_prob  # Get all predictive choice probability [K, num_trials]
        likelihood_each_trial = predictive_choice_prob [choice_this[0,:], range(len(choice_this[0]))]  # Get the actual likelihood for each trial
        
        # Deal with numerical precision
        likelihood_each_trial[(likelihood_each_trial <= 0) & (likelihood_each_trial > -1e-5)] = 1e-16  # To avoid infinity, which makes the number of zero likelihoods informative!
        likelihood_each_trial[likelihood_each_trial > 1] = 1
        
        # Cache likelihoods
        likelihood_all_trial.extend(likelihood_each_trial)
    
    likelihood_all_trial = np.array(likelihood_all_trial)
    
    if fit_set == []: # Use all trials
        negLL = - sum(np.log(likelihood_all_trial))
    else:   # Only return likelihoods in the fit_set
        negLL = - sum(np.log(likelihood_all_trial[fit_set]))
        
    # print(np.round(fit_value,4), negLL, '\n')
    # if np.any(likelihood_each_trial < 0):
    #     print(predictive_choice_prob)
    
    return negLL



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




# %%
from typing import Literal, Tuple, Type

from pydantic import BaseModel, Field

# from .forager_q_learning_params import _add_choice_kernel_fields
# from .util import create_pydantic_models_dynamic


def generate_pydantic_loss_counting_params(
    win_stay_lose_switch: Literal[False, True] = False,
    choice_kernel: Literal["none", "one_step", "full"] = "none",
) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Generate Pydantic models for Loss-counting agent parameters.

    All default values are hard-coded in this function. But when instantiating the model,
    you can always override the default values, both the params and the fitting bounds.

    Parameters
    ----------
    win_stay_lose_switch : bool, optional
        If True, the agent will be a win-stay-lose-shift agent
        (loss_count_threshold_mean and loss_count_threshold_std are fixed at 1 and 0),
        by default False
    choice_kernel : Literal["none", "one_step", "full"], optional
        Choice kernel type, by default "none"
        If "none", no choice kernel will be included in the model.
        If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the previous choice
            affects the choice kernel. (Bari2019)
        If "full", both choice_kernel_step_size and choice_kernel_relative_weight will be included
    """

    # ====== Define common fields and constraints ======
    params_fields = {}
    fitting_bounds = {}

    # -- Loss counting model parameters --
    if win_stay_lose_switch:
        params_fields["loss_count_threshold_mean"] = (
            float,
            Field(
                default=1.0,
                ge=1.0,
                le=1.0,
                frozen=True,  # To indicate that this field is clamped by construction
                description="Mean of the loss count threshold",
            ),
        )
        fitting_bounds["loss_count_threshold_mean"] = (1.0, 1.0)

        params_fields["loss_count_threshold_std"] = (
            float,
            Field(
                default=0.0,
                ge=0.0,
                le=0.0,
                frozen=True,  # To indicate that this field is clamped by construction
                description="Std of the loss count threshold",
            ),
        )
        fitting_bounds["loss_count_threshold_std"] = (0.0, 0.0)
    else:
        params_fields["loss_count_threshold_mean"] = (
            float,
            Field(
                default=1.0,
                ge=0.0,
                description="Mean of the loss count threshold",
            ),
        )
        fitting_bounds["loss_count_threshold_mean"] = (0.0, 10.0)

        params_fields["loss_count_threshold_std"] = (
            float,
            Field(
                default=0.0,
                ge=0.0,
                description="Std of the loss count threshold",
            ),
        )
        fitting_bounds["loss_count_threshold_std"] = (0.0, 10.0)

    # -- Always add a bias term --
    params_fields["biasL"] = (
        float,
        Field(default=0.0, ge=-1.0, le=1.0, description="Bias term for loss counting"),
    )  # Bias term for loss counting directly added to the choice probabilities
    fitting_bounds["biasL"] = (-1.0, 1.0)

    # -- Add choice kernel fields --
    _add_choice_kernel_fields(params_fields, fitting_bounds, choice_kernel)

    return create_pydantic_models_dynamic(params_fields, fitting_bounds)


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



def act_loss_counting(
    previous_choice: Optional[int],
    loss_count: int,
    loss_count_threshold_mean: float,
    loss_count_threshold_std: float,
    bias_terms: np.array,
    choice_kernel=None,
    choice_kernel_relative_weight=None,
    rng=None,
):
    """Action selection by loss counting method.

    Parameters
    ----------
    previous_choice : int
        Last choice
    loss_count : int
        Current loss count
    loss_count_threshold_mean : float
        Mean of the loss count threshold
    loss_count_threshold_std : float
        Standard deviation of the loss count threshold
    bias_terms: np.array
        Bias terms loss count
    choice_kernel : None or np.array, optional
        If not None, it will be added to Q-values, by default None
    choice_kernel_relative_weight : _type_, optional
        If not None, it controls the relative weight of choice kernel, by default None
    rng : _type_, optional
    """
    rng = rng or np.random.default_rng()

    # -- Return random if this is the first trial --
    if previous_choice is None:
        choice_prob = np.array([0.5, 0.5])
        return choose_ps(choice_prob, rng=rng), choice_prob

    # -- Compute probability of switching --
    # This cdf trick is equivalent to:
    #   1) sample a threshold from the normal distribution
    #   2) compare the threshold with the loss count
    prob_switch = norm.cdf(
        loss_count,
        loss_count_threshold_mean
        - 1e-10,  # To make sure this is equivalent to ">=" if the threshold is an integer
        loss_count_threshold_std + 1e-16,  # To make sure this cdf trick works for std=0
    )
    choice_prob = np.array([prob_switch, prob_switch])  # Assuming only two choices
    choice_prob[int(previous_choice)] = 1 - prob_switch

    # -- Add choice kernel --
    # For a fair comparison with other models that have choice kernel.
    # However, choice kernel of different families are not directly comparable.
    # Here, I first compute a normalized choice probability for choice kernel alone using softmax
    # with inverse temperature 1.0, compute a bias introduced by the choice kernel, and then add
    # it to the original choice probability.
    if choice_kernel is not None:
        choice_prob_choice_kernel = softmax(choice_kernel, rng=rng)
        bias_L_from_choice_kernel = (
            choice_prob_choice_kernel[L] - 0.5
        ) * choice_kernel_relative_weight  # A biasL term introduced by the choice kernel
        choice_prob[L] += bias_L_from_choice_kernel

    # -- Add global bias --
    # For a fair comparison with other models that have bias terms.
    # However, bias terms of different families are not directly comparable.
    # Here, the bias term is added to the choice probability directly, whereas in other models,
    # the bias term is added to the Q-values.
    choice_prob[L] += bias_terms[L]

    # -- Re-normalize choice probability --
    choice_prob[L] = np.clip(choice_prob[L], 0, 1)
    choice_prob[R] = 1 - choice_prob[L]

    return choose_ps(choice_prob, rng=rng), choice_prob



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


def learn_loss_counting(choice, reward, just_switched, loss_count_tminus1) -> int:
    """Update loss counting

    Returns the new loss count
    """
    if reward:
        return 0

    # If not reward
    if just_switched:
        return 1
    else:
        return loss_count_tminus1 + 1



"""Maximum likelihood fitting of foraging models"""

# %%
from typing import Literal
from aind_behavior_gym.dynamic_foraging.task import L, R



class ForagerLossCounting(DynamicForagingAgentMLEBase):
    """The familiy of loss counting models."""

    def __init__(
        self,
        win_stay_lose_switch: Literal[False, True] = False,
        choice_kernel: Literal["none", "one_step", "full"] = "none",
        params: dict = {},
        **kwargs,
    ):
        """Initialize the family of loss counting agents.

        Some special agents are:
        1. Never switch: loss_count_threshold_mean = inf
        2. Always switch: loss_count_threshold_mean = 0.0 & loss_count_threshold_std = 0.0
        3. Win-stay-lose-shift: loss_count_threshold_mean = 1.0 & loss_count_threshold_std = 0.0

        Parameters
        ----------
        win_stay_lose_switch: bool, optional
            If True, the agent will be a win-stay-lose-shift agent
            (loss_count_threshold_mean and loss_count_threshold_std are fixed at 1 and 0),
            by default False
        choice_kernel : Literal["none", "one_step", "full"], optional
            Choice kernel type, by default "none"
            If "none", no choice kernel will be included in the model.
            If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the last choice
                affects the choice kernel. (Bari2019)
            If "full", both choice_kernel_step_size and choice_kernel_relative_weight
            will be included in fitting
        params: dict, optional
            Initial parameters of the model, by default {}.
            In the loss counting model, the only two parameters are:
                - loss_count_threshold_mean: float
                - loss_count_threshold_std: float
        """
        # -- Pack the agent_kwargs --
        self.agent_kwargs = dict(
            win_stay_lose_switch=win_stay_lose_switch,
            choice_kernel=choice_kernel,
        )

        # -- Initialize the model parameters --
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

        # -- Some agent-family-specific variables --
        self.fit_choice_kernel = False

    def _get_params_model(self, agent_kwargs):
        """Get the params model of the agent"""
        return generate_pydantic_loss_counting_params(**agent_kwargs)

    def get_agent_alias(self):
        """Get the agent alias"""
        _prefix = "WSLS" if self.agent_kwargs["win_stay_lose_switch"] else "LossCounting"
        _ck = {"none": "", "one_step": "_CK1", "full": "_CKfull"}[
            self.agent_kwargs["choice_kernel"]
        ]
        return _prefix + _ck

    def _reset(self):
        """Reset the agent"""
        # --- Call the base class reset ---
        super()._reset()

        # --- Agent family specific variables ---
        self.loss_count = np.full(self.n_trials + 1, np.nan)
        self.loss_count[0] = 0  # Initial loss count as 0

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

        choice, choice_prob = act_loss_counting(
            previous_choice=self.choice_history[self.trial - 1] if self.trial > 0 else None,
            loss_count=self.loss_count[self.trial],
            loss_count_threshold_mean=self.params.loss_count_threshold_mean,
            loss_count_threshold_std=self.params.loss_count_threshold_std,
            bias_terms=np.array([self.params.biasL, 0]),
            # -- Choice kernel --
            choice_kernel=choice_kernel,
            choice_kernel_relative_weight=choice_kernel_relative_weight,
            rng=self.rng,
        )
        return choice, choice_prob

    def learn(self, _, choice, reward, __, done):
        """Update loss counter

        Note that self.trial already increased by 1 before learn() in the base class
        """
        self.loss_count[self.trial] = learn_loss_counting(
            choice=choice,
            reward=reward,
            just_switched=(self.trial == 1 or choice != self.choice_history[self.trial - 2]),
            loss_count_tminus1=self.loss_count[self.trial - 1],
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
            "loss_count": self.loss_count.tolist(),
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

        if not if_fitted:
            # Only plot loss count if not fitted
            ax_loss_count = ax.twinx()

            ax_loss_count.plot(x, self.loss_count, label="loss_count", color="blue", **style)
            ax_loss_count.set(ylabel="Loss count")
            ax_loss_count.legend(loc="upper right", fontsize=6)

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




class RandomWalkReward:
    '''
    Generate reward schedule with random walk

    (see Miller et al. 2021, https://www.biorxiv.org/content/10.1101/461129v3.full.pdf)
    '''

    def __init__(self,
                 p_min=[0, 0],  # L and R
                 p_max=[1, 1], # L and R 
                 sigma=[0.15, 0.15],  # L and R
                 mean=[0, 0],         # L and R
                 ) -> None:
        
        self.__dict__.update(locals())
        
        if not type(sigma) == list:
            sigma = [sigma, sigma]  # Backward compatibility
            
        if not type(p_min) == list:
            p_min = [p_min, p_min]  # Backward compatibility

        if not type(p_max) == list:
            p_max = [p_max, p_max]  # Backward compatibility
       
        if not type(mean) == list:
            mean = [mean, mean]  # Backward compatibility

        self.p_min, self.p_max, self.sigma, self.mean = p_min, p_max, sigma, mean
                   
        self.trial_rwd_prob = {'L':[], 'R': []}  # Rwd prob per trial
        self.choice_history = []

        self.hold_this_block = False
        self.first_trial()
    
    def first_trial(self): 
        self.trial_now = 0
        for i, side in enumerate(['L', 'R']):
            self.trial_rwd_prob[side].append(np.random.uniform(self.p_min[i], self.p_max[i]))
            
    def next_trial(self):
        self.trial_now += 1
        for i, side in enumerate(['L', 'R']):
            if not self.hold_this_block:
                p = np.random.normal(self.trial_rwd_prob[side][-1] + self.mean[i], self.sigma[i])
                p = min(self.p_max[i], max(self.p_min[i], p))
            else:
                p = self.trial_rwd_prob[side][-1]
            self.trial_rwd_prob[side].append(p)

    def add_choice(self, this_choice):
        self.choice_history.append(this_choice)

    def auto_corr(self, data):
        mean = np.mean(data)
        # Variance
        var = np.var(data)
        # Normalized data
        ndata = data - mean
        acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
        acorr = acorr / var / len(ndata)
        return acorr

    def plot_reward_schedule(self):
        fig, ax = plt.subplots(2, 2, figsize=[15, 7], sharex='col', gridspec_kw=dict(width_ratios=[4, 1], wspace=0.1))

        for s, col in zip(['L', 'R'], ['r', 'b']):
            ax[0, 0].plot(self.trial_rwd_prob[s], col, marker='.', alpha=0.5, lw=2)
            ax[0, 1].plot(self.auto_corr(self.trial_rwd_prob[s]), col)

        ax[1, 0].plot(np.array(self.trial_rwd_prob['L']) + np.array(self.trial_rwd_prob['R']), label='sum')
        ax[1, 0].plot(np.array(self.trial_rwd_prob['R']) / (np.array(self.trial_rwd_prob['L']) + np.array(self.trial_rwd_prob['R'])), label='R/(L+R)')
        ax[1, 0].legend()

        ax[0, 1].set(title='auto correlation', xlim=[0, 100])
        ax[0, 1].axhline(y=0, c='k', ls='--')

        plt.show()
        fig.savefig('results/random_walk.png')
        


from scipy.stats import norm

LEFT = 0
RIGHT = 1

global_block_size_mean = 80
global_block_size_sd = 20


class BanditModel:
    '''
    Foragers that can simulate and fit bandit models
    '''

    # @K_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations

    def __init__(self, forager=None, K_arm=2, n_trials=1000, if_baited=True,
                 
                 if_para_optim=False,
                 if_varying_amplitude=False,

                 epsilon=None,               # For 'RW1972_epsi'
                 # For 'LNP_softmax', 'RW1972_softmax', 'Bari2019', 'Hattori2019'
                 softmax_temperature=None,

                 # Bias terms, K-1 degrees of freedom, with constraints:
                 # 1. for those involve random:  b_K = 0 - sum_1toK-1_(b_k), -1/K < b_k < (K-1)/K. cp_k = cp + b_k (for pMatching, may be truncated)
                 # 2. for those involve softmax: b_K = 0, no constraint. cp_k = exp(Q/sigma + b_k) / sum(Q/sigma + b_k). Putting b_k outside /sigma to make it comparable across different softmax_temperatures
                 biasL=0,  # For K = 2.
                 biasR=0,  # Only for K = 3

                 # For 'LNP_softmax', up to two taus
                 tau1=None,
                 tau2=None,
                 w_tau1=None,

                 # Choice kernel
                 choice_step_size=None,
                 choice_softmax_temperature=None,

                 # For 'RW1972_epsi','RW1972_softmax','Bari2019', 'Hattori2019'
                 learn_rate=None,  # For RW and Bari
                 learn_rate_rew=None,     # For Hattori
                 learn_rate_unrew=None,     # For Hattori
                 # 'RW1972_xxx' (= 0)， 'Bari2019' (= 1-Zeta)， 'Hattori2019' ( = unchosen_forget_rate).
                 forget_rate=None,

                 # For 'LossCounting' [from Shahidi 2019]
                 loss_count_threshold_mean=None,
                 loss_count_threshold_std=0,

                 # If true, use the same random seed for generating p_reward!!
                 p_reward_seed_override='',
                 p_reward_sum=0.45,   # Gain of reward. Default = 0.45
                 p_reward_pairs=None,  # Full control of reward prob

                 # !! Important for predictive fitting !!
                 # If not None, calculate predictive_choice_probs(t) based on fit_choice_history(0:t-1) and fit_reward_history(0:t-1) for negLL calculation.
                 fit_choice_history=None,
                 fit_reward_history=None,
                 fit_iti=None,  # ITI [t] --> ITI between t and t + 1

                 # For CANN
                 tau_cann=None,
                 
                 # For synaptic network
                 rho = None,
                 I0 = None,
                 ):

        self.forager = forager
        self.if_baited = if_baited
        self.if_varying_amplitude = if_varying_amplitude
        self.if_para_optim = if_para_optim

        self.epsilon = epsilon
        self.softmax_temperature = softmax_temperature
        self.loss_count_threshold_mean = loss_count_threshold_mean
        self.loss_count_threshold_std = loss_count_threshold_std
        self.p_reward_seed_override = p_reward_seed_override
        self.p_reward_sum = p_reward_sum
        self.p_reward_pairs = p_reward_pairs

        self.fit_choice_history = fit_choice_history
        self.fit_reward_history = fit_reward_history
        self.iti = fit_iti
        # In some cases we just need fit_c to fit the model
        self.if_fit_mode = self.fit_choice_history is not None

        if self.if_fit_mode:
            self.K, self.n_trials = np.shape(
                fit_reward_history)  # Use the targeted histories

        else:  # Backward compatibility
            self.K = K_arm
            self.n_trials = n_trials
        
        self.description = f'{self.forager}'
        self.task = 'Bandit'

        # =============================================================================
        #   Parameter check and prepration
        # =============================================================================

        # -- Bias terms --
        # K-1 degrees of freedom, with constraints:
        # 1. for those involve random:  sum_(b_k) = 0, -1/K < b_k < (K-1)/K. cp_k = cp + b_k (for pMatching, may be truncated)
        if forager in ['Random', 'pMatching', 'RW1972_epsi']:
            if self.K == 2:
                self.bias_terms = np.array(
                    [biasL, -biasL])  # Relative to right
            elif self.K == 3:
                self.bias_terms = np.array(
                    [biasL, -(biasL + biasR), biasR])  # Relative to middle
            # Constraints (no need)
            # assert np.all(-1/self.K <= self.bias_terms) and np.all(self.bias_terms <= (self.K - 1)/self.K), self.bias_terms

        # 2. for those involve softmax: b_undefined = 0, no constraint. cp_k = exp(Q/sigma + b_i) / sum(Q/sigma + b_i). Putting b_i outside /sigma to make it comparable across different softmax_temperatures
        elif forager in ['RW1972_softmax', 'LNP_softmax', 'LNP_epsi', 'Bari2019', 'Hattori2019',
                         'RW1972_softmax_CK', 'LNP_softmax_CK', 'Bari2019_CK', 'Hattori2019_CK',
                         'CANN', 'Synaptic', 'Synaptic_W>0']:
            if self.K == 2:
                self.bias_terms = np.array([biasL, 0])  # Relative to right
            elif self.K == 3:
                self.bias_terms = np.array(
                    [biasL, 0, biasR])  # Relative to middle
            # No constraints

        # -- Forager-dependent --
        if 'LNP' in forager:
            assert all(x is not None for x in (tau1,))
            if tau2 == None:  # Only one tau ('Sugrue2004')
                self.taus = [tau1]
                self.w_taus = [1]
            else:                           # 'Corrado2005'
                self.taus = [tau1, tau2]
                self.w_taus = [w_tau1, 1 - w_tau1]
                
            self.description += ', taus = %s, w_taus = %s' % \
                                (np.round(self.taus,3), np.round(self.w_taus,3))

        elif 'RW1972' in forager:
            assert all(x is not None for x in (learn_rate,))
            # RW1972 has the same learning rate for rewarded / unrewarded trials
            self.learn_rates = [learn_rate, learn_rate]
            self.forget_rates = [0, 0]   # RW1972 does not forget
            
            self.description += ', learn rate = %s' % (np.round(learn_rate, 3)) 

        elif 'Bari2019' in forager:
            assert all(x is not None for x in (learn_rate, forget_rate))
            # Bari2019 also has the same learning rate for rewarded / unrewarded trials
            self.learn_rates = [learn_rate, learn_rate]
            self.forget_rates = [forget_rate, forget_rate]
            
            self.description += ', learn_rate = %s, forget_rate = %s' % \
                       (np.round(learn_rate, 3), np.round(forget_rate, 3))

        elif 'Hattori2019' in forager:
            assert all(x is not None for x in (
                learn_rate_rew, learn_rate_unrew))
            if forget_rate is None:
                # Allow Hattori2019 to not have forget_rate. In that case, it is an extension of RW1972.
                forget_rate = 0

            # 0: unrewarded, 1: rewarded
            self.learn_rates = [learn_rate_unrew, learn_rate_rew]
            self.forget_rates = [forget_rate, 0]   # 0: unchosen, 1: chosen
            
            self.description += ', learn_rates (unrew, rew) = %s, forget_rate = %s' %\
                (np.round(self.learn_rates, 3), np.round(forget_rate, 3))

        elif 'CANN' in forager:
            assert all(x is not None for x in (
                learn_rate, tau_cann, softmax_temperature))
            self.tau_cann = tau_cann
            self.learn_rates = [learn_rate, learn_rate]
            
        elif 'Synaptic' in forager:
            assert all(x is not None for x in (
                learn_rate, forget_rate, I0, rho, softmax_temperature))
            self.I0 = I0
            self.rho = rho
            self.learn_rates = [learn_rate, learn_rate]
            self.forget_rates = [forget_rate, forget_rate]
            
        if any([x in forager for x in ('softmax', 'Bari2019', 'Hattori2019')]):
            assert all(x is not None for x in (self.softmax_temperature,))
            self.description += ', softmax_temp = %s' % (np.round(self.softmax_temperature, 3))

        if 'epsi' in forager:
            assert all(x is not None for x in (self.epsilon,))
            self.description += ', epsilon = %s'  % (np.round(self.epsilon, 3))
                    
        # Choice kernel can be added to any reward-based forager
        if '_CK' in forager:
            assert choice_step_size is not None and choice_softmax_temperature is not None
            self.choice_step_size = choice_step_size
            self.choice_softmax_temperature = choice_softmax_temperature
            
            self.description += ', choice_kernel_step_size = %s, choice_softmax_temp = %s' %\
                (np.round(choice_step_size, 3), np.round(choice_softmax_temperature, 3))


    def reset(self):

        #  print(self)

        # Initialization
        self.time = 0

        # All latent variables have n_trials + 1 length to capture the update after the last trial (HH20210726)
        self.q_estimation = np.full([self.K, self.n_trials + 1], np.nan)
        self.q_estimation[:, 0] = 0

        self.choice_prob = np.full([self.K, self.n_trials + 1], np.nan)
        self.choice_prob[:, 0] = 1/self.K   # To be strict (actually no use)

        if self.if_fit_mode:  # Predictive mode
            self.predictive_choice_prob = np.full(
                [self.K, self.n_trials + 1], np.nan)
            # To be strict (actually no use)
            self.predictive_choice_prob[:, 0] = 1/self.K

        else:   # Generative mode
            self.choice_history = np.zeros(
                [1, self.n_trials + 1], dtype=int)  # Choice history
            # Reward history, separated for each port (Corrado Newsome 2005)
            self.reward_history = np.zeros([self.K, self.n_trials + 1])

            # Generate baiting prob in block structure
            self.generate_p_reward()

            # Prepare reward for the first trial
            # For example, [0,1] represents there is reward baited at the RIGHT but not LEFT port.
            # Reward history, separated for each port (Corrado Newsome 2005)
            self.reward_available = np.zeros([self.K, self.n_trials + 1])
            self.reward_available[:, 0] = (np.random.uniform(
                0, 1, self.K) < self.p_reward[:, self.time]).astype(int)

        # Forager-specific
        if self.forager in ['RW1972_epsi', 'RW1972_softmax', 'Bari2019', 'Hattori2019',
                            'RW1972_softmax_CK', 'Bari2019_CK', 'Hattori2019_CK']:
            pass

        elif 'LNP' in self.forager:   # 'LNP_softmax', 'LNP_softmax_CK', 'LNP_epsi'
            # Compute the history filter. Compatible with any number of taus.
            # Use the full length of the session just in case of an extremely large tau.
            reversed_t = np.flipud(np.arange(self.n_trials + 1))
            self.history_filter = np.zeros_like(reversed_t).astype('float64')

            for tau, w_tau in zip(self.taus, self.w_taus):
                # Note the normalization term (= tau when n -> inf.)
                self.history_filter += w_tau * \
                    np.exp(-reversed_t / tau) / \
                    np.sum(np.exp(-reversed_t / tau))

        elif self.forager in ['LossCounting']:
            # Initialize
            self.loss_count = np.zeros([1, self.n_trials + 1])
            if not self.if_fit_mode:
                self.loss_threshold_this = np.random.normal(
                    self.loss_count_threshold_mean, self.loss_count_threshold_std)

        elif 'CANN' in self.forager:
            if not self.if_fit_mode:   # Override user input of iti
                self.iti = np.ones(self.n_trials)
                
        elif 'Synaptic' in self.forager:
            self.w = np.full([self.K, self.n_trials + 1], np.nan)
            self.w[:, 0] = 0.1

        # Choice kernel can be added to any forager
        if '_CK' in self.forager:
            self.choice_kernel = np.zeros([self.K, self.n_trials + 1])

    def generate_p_reward(self, block_size_base=global_block_size_mean,
                          block_size_sd=global_block_size_sd,
                          # (Bari-Cohen 2019)
                          p_reward_pairs=[
                              [.4, .05], [.3857, .0643], [.3375, .1125], [.225, .225]],
                          ):

        # If para_optim, fix the random seed to ensure that p_reward schedule is fixed for all candidate parameters
        # However, we should make it random during a session (see the last line of this function)
        if self.p_reward_seed_override != '':
            np.random.seed(self.p_reward_seed_override)

        if self.p_reward_pairs == None:
            p_reward_pairs = np.array(
                p_reward_pairs) / 0.45 * self.p_reward_sum
        else:  # Full override of p_reward
            p_reward_pairs = self.p_reward_pairs

        # Adapted from Marton's code
        n_trials_now = 0
        block_size = []
        n_trials = self.n_trials + 1
        p_reward = np.zeros([2, n_trials])
        
        self.rewards_IdealpHatOptimal = 0
        self.rewards_IdealpHatGreedy = 0


        # Fill in trials until the required length
        while n_trials_now < n_trials:

            # Number of trials in each block (Gaussian distribution)
            # I treat p_reward[0,1] as the ENTIRE lists of reward probability. RIGHT = 0, LEFT = 1. HH
            n_trials_this_block = np.rint(np.random.normal(
                block_size_base, block_size_sd)).astype(int)
            n_trials_this_block = min(
                n_trials_this_block, n_trials - n_trials_now)

            block_size.append(n_trials_this_block)

            # Get values to fill for this block
            # If 0, the first block is set to 50% reward rate (as Marton did)
            if n_trials_now == -1:
                p_reward_this_block = np.array(
                    [[sum(p_reward_pairs[0])/2] * 2])  # Note the outer brackets
            else:
                # Choose reward_ratio_pair
                # If we had equal p_reward in the last block
                if n_trials_now > 0 and not(np.diff(p_reward_this_block)):
                    # We should not let it happen again immediately
                    pair_idx = np.random.choice(range(len(p_reward_pairs)-1))
                else:
                    pair_idx = np.random.choice(range(len(p_reward_pairs)))

                p_reward_this_block = np.array(
                    [p_reward_pairs[pair_idx]])   # Note the outer brackets

                # To ensure flipping of p_reward during transition (Marton)
                if len(block_size) % 2:
                    p_reward_this_block = np.flip(p_reward_this_block)

            # Fill in trials for this block
            p_reward[:, n_trials_now: n_trials_now +
                     n_trials_this_block] = p_reward_this_block.T

            # Fill choice history for some special foragers with choice patterns {AmBn} (including IdealpHatOptimal, IdealpHatGreedy, and AmB1)
            self.get_AmBn_choice_history(p_reward_this_block, n_trials_this_block, n_trials_now)
            
            # Next block
            n_trials_now += n_trials_this_block

        self.n_blocks = len(block_size)
        self.p_reward = p_reward
        self.block_size = np.array(block_size)
        self.p_reward_fraction = p_reward[RIGHT, :] / \
            (np.sum(p_reward, axis=0))   # For future use
        self.p_reward_ratio = p_reward[RIGHT, :] / \
            p_reward[LEFT, :]   # For future use

        # We should make it random afterwards
        np.random.seed()
        

    def get_AmBn_choice_history(self, p_reward_this_block, n_trials_this_block, n_trials_now):
        
        if not self.if_para_optim:  
            # Calculate theoretical upper bound (ideal-p^-optimal) and the (fixed) choice history/matching point of it
            # Ideal-p^-Optimal
            # mn_star_pHatOptimal, p_star_pHatOptimal = self.get_IdealpHatOptimal_strategy(p_reward_this_block[0])
            # self.rewards_IdealpHatOptimal += p_star_pHatOptimal * n_trials_this_block
            pass

        # Ideal-p^-Greedy
        mn_star_pHatGreedy, p_star_pHatGreedy = self.get_IdealpHatGreedy_strategy(
            p_reward_this_block[0])
        mn_star = mn_star_pHatGreedy
        # Ideal-p^-Greedy
        self.rewards_IdealpHatGreedy += p_star_pHatGreedy * n_trials_this_block


        if self.forager == 'IdealpHatGreedy':
            # For ideal optimal, given p_0(t) and p_1(t), the optimal choice history is fixed, i.e., {m_star, 1} (p_min > 0)
            S = int(np.ceil(n_trials_this_block/(mn_star[0] + mn_star[1])))
            c_max_this = np.argwhere(p_reward_this_block[0] == np.max(
                p_reward_this_block))[0]  # To handle the case of p0 = p1
            c_min_this = np.argwhere(
                p_reward_this_block[0] == np.min(p_reward_this_block))[-1]
            # Choice pattern of {m_star, 1}
            c_star_this_block = ([c_max_this] * mn_star[0] +
                                [c_min_this] * mn_star[1]) * S
            # Truncate to the correct length
            c_star_this_block = c_star_this_block[:n_trials_this_block]

            self.choice_history[0, n_trials_now: n_trials_now +
                                n_trials_this_block] = np.hstack(c_star_this_block)  # Save the optimal sequence
            

    def get_IdealpHatGreedy_strategy(self, p_reward):
        '''
        Ideal-p^-greedy, only care about the current p^, which is good enough (for 2-arm task)  03/28/2020
        '''
        p_max = np.max(p_reward)
        p_min = np.min(p_reward)

        if p_min > 0:
            m_star = np.floor(np.log(1-p_max)/np.log(1-p_min))
            p_star = p_max + (1-(1-p_min)**(m_star + 1)-p_max**2) / \
                (m_star+1)  # Still stands even m_star = *

            return [int(m_star), 1], p_star
        else:
            # Safe to be always on p_max side for this block
            return [self.n_trials, 1], p_max

    def act_random(self):

        if self.if_fit_mode:
            self.predictive_choice_prob[:,
                                        self.time] = 1/self.K + self.bias_terms
            choice = None   # No need to make specific choice in fitting mode
        else:
            # choice = np.random.choice(self.K)
            choice = choose_ps(1/self.K + self.bias_terms)
            self.choice_history[0, self.time] = choice
        return choice

    def act_LossCounting(self):

        if self.time == 0:  # Only this need special initialization
            if self.if_fit_mode:
                # No need to update self.predictive_choice_prob[:, self.time]
                pass
                return None
            else:
                return np.random.choice(self.K)

        if self.if_fit_mode:
            # Retrieve the last choice
            last_choice = self.fit_choice_history[0, self.time - 1]

            # Predict this choice prob
            # To be general, and ensure that alway switch when mean = 0, std = 0
            prob_switch = norm.cdf(
                self.loss_count[0, self.time], self.loss_count_threshold_mean - 1e-6, self.loss_count_threshold_std + 1e-16)

            # Choice prob [choice] = 1-prob_switch, [others] = prob_switch /(K-1). Assuming randomly switch to other alternatives
            self.predictive_choice_prob[:,
                                        self.time] = prob_switch / (self.K - 1)
            self.predictive_choice_prob[last_choice,
                                        self.time] = 1 - prob_switch

            # Using fit_choice to mark an actual switch
            if self.time < self.n_trials and last_choice != self.fit_choice_history[0, self.time]:
                # A flag of "switch happens here"
                self.loss_count[0, self.time] = - self.loss_count[0, self.time]

            choice = None

        else:
            # Retrieve the last choice
            last_choice = self.choice_history[0, self.time - 1]

            if self.loss_count[0, self.time] >= self.loss_threshold_this:
                # Switch
                choice = LEFT + RIGHT - last_choice

                # Reset loss counter threshold
                # A flag of "switch happens here"
                self.loss_count[0, self.time] = - self.loss_count[0, self.time]
                self.loss_threshold_this = np.random.normal(
                    self.loss_count_threshold_mean, self.loss_count_threshold_std)
            else:
                # Stay
                choice = last_choice

            self.choice_history[0, self.time] = choice

        return choice

    def act_EpsiGreedy(self):

        # if np.random.rand() < self.epsilon:
        #     # Forced exploration with the prob. of epsilon (to avoid AlwaysLEFT/RIGHT in Sugrue2004...) or before some rewards are collected
        #     choice = self.act_random()

        # else:    # Greedy
        #     choice = np.random.choice(np.where(self.q_estimation[:, self.time] == self.q_estimation[:, self.time].max())[0])
        #     if self.if_fit_mode:
        #         self.predictive_choice_prob[:, self.time] = 0
        #         self.predictive_choice_prob[choice, self.time] = 1  # Delta-function
        #         choice = None   # No need to make specific choice in fitting mode
        #     else:
        #         self.choice_history[0, self.time] = choice

        # == The above is erroneous!! We should never realize any probabilistic events in model fitting!! ==
        choice = np.random.choice(np.where(
            self.q_estimation[:, self.time] == self.q_estimation[:, self.time].max())[0])

        if self.if_fit_mode:
            self.predictive_choice_prob[:, self.time] = self.epsilon * \
                (1 / self.K + self.bias_terms)
            self.predictive_choice_prob[choice, self.time] = 1 - self.epsilon + \
                self.epsilon * (1 / self.K + self.bias_terms[choice])
            choice = None   # No need to make specific choice in fitting mode
        else:
            if np.random.rand() < self.epsilon:
                choice = self.act_random()

            self.choice_history[0, self.time] = choice

        return choice

    def act_Probabilistic(self):

        # !! Should not change q_estimation!! Otherwise will affect following Qs
        # And I put softmax here
        if '_CK' in self.forager:
            self.choice_prob[:, self.time] = softmax(np.vstack([self.q_estimation[:, self.time], self.choice_kernel[:, self.time]]),
                                                     np.vstack(
                                                         [self.softmax_temperature, self.choice_softmax_temperature]),
                                                     bias=self.bias_terms)  # Updated softmax function that accepts two elements
        else:
            self.choice_prob[:, self.time] = softmax(
                self.q_estimation[:, self.time], self.softmax_temperature, bias=self.bias_terms)

        if self.if_fit_mode:
            self.predictive_choice_prob[:,
                                        self.time] = self.choice_prob[:, self.time]
            choice = None   # No need to make specific choice in fitting mode
        else:
            choice = choose_ps(self.choice_prob[:, self.time])
            self.choice_history[0, self.time] = choice

        return choice

    def step_LossCounting(self, reward):

        if self.loss_count[0, self.time - 1] < 0:  # A switch just happened
            # Back to normal (Note that this = 0 in Shahidi 2019)
            self.loss_count[0, self.time - 1] = - \
                self.loss_count[0, self.time - 1]
            if reward:
                self.loss_count[0, self.time] = 0
            else:
                self.loss_count[0, self.time] = 1
        else:
            if reward:
                self.loss_count[0,
                                self.time] = self.loss_count[0, self.time - 1]
            else:
                self.loss_count[0, self.time] = self.loss_count[0,
                                                                self.time - 1] + 1

    def step_LNP(self, valid_reward_history):

        valid_filter = self.history_filter[-self.time:]
        local_income = np.sum(valid_reward_history * valid_filter, axis=1)

        self.q_estimation[:, self.time] = local_income

    def step_RWlike(self, choice, reward):

        # Reward-dependent step size ('Hattori2019')
        if reward:
            learn_rate_this = self.learn_rates[1]
        else:
            learn_rate_this = self.learn_rates[0]

        # Choice-dependent forgetting rate ('Hattori2019')
        # Chosen:   Q(n+1) = (1- forget_rate_chosen) * Q(n) + step_size * (Reward - Q(n))
        self.q_estimation[choice, self.time] = (1 - self.forget_rates[1]) * self.q_estimation[choice, self.time - 1]  \
            + learn_rate_this * \
            (reward - self.q_estimation[choice, self.time - 1])

        # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
        unchosen_idx = [cc for cc in range(self.K) if cc != choice]
        self.q_estimation[unchosen_idx, self.time] = (
            1 - self.forget_rates[0]) * self.q_estimation[unchosen_idx, self.time - 1]

        # --- The below three lines are erroneous!! Should not change q_estimation!! ---
        # Softmax in 'Bari2019', 'Hattori2019'
        # if self.forager in ['RW1972_softmax', 'Bari2019', 'Hattori2019']:
        #     self.q_estimation[:, self.time] = softmax(self.q_estimation[:, self.time], self.softmax_temperature)

    def step_CANN(self, choice, reward):
        """
        Abstracted from Ulises' line attractor model
        """
        if reward:
            learn_rate_this = self.learn_rates[1]
        else:
            learn_rate_this = self.learn_rates[0]
            
        # ITI[self.time] --> ITI between (self.time) and (self.time + 1)
        iti_time_minus1_to_time = self.iti[self.time - 1]

        # Choice-dependent forgetting rate ('Hattori2019')
        # Chosen:   Q(n+1) = (1- forget_rate_chosen) * Q(n) + step_size * (Reward - Q(n))
        self.q_estimation[choice, self.time] = (self.q_estimation[choice, self.time - 1]
                                                + learn_rate_this * (reward - self.q_estimation[choice, self.time - 1])
                                                ) * np.exp( -iti_time_minus1_to_time / self.tau_cann)

        # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
        unchosen_idx = [cc for cc in range(self.K) if cc != choice]
        self.q_estimation[unchosen_idx, self.time] = self.q_estimation[unchosen_idx, self.time - 1
                                                                       ] * np.exp( -iti_time_minus1_to_time / self.tau_cann)
        
    @staticmethod
    def f(x): 
        return 0 if x <= 0 else 1 if x >= 1 else x

    def step_synaptic(self, choice, reward):
        """
        Abstracted from Ulises' mean-field synaptic model
        """
        # -- Update w --
        if reward:
            learn_rate_this = self.learn_rates[1]
        else:
            learn_rate_this = self.learn_rates[0]
            
        # Chosen side
        self.w[choice, self.time] = (1 - self.forget_rates[1]) * self.w[choice, self.time - 1] \
            + learn_rate_this * (reward - self.q_estimation[choice, self.time - 1]) * self.q_estimation[choice, self.time - 1]
        # Unchosen side
        self.w[1 - choice, self.time] = (1 - self.forget_rates[0]) * self.w[1 - choice, self.time - 1]
        
        # Rectify w if needed
        if self.forager == 'Synaptic_W>0':
            self.w[self.w[:, self.time] < 0, self.time] = 0
        
        # -- Update u --
        for side in [0, 1]:
            self.q_estimation[side, self.time] = self.f(self.I0 * (1 - self.w[1 - side, self.time]) / 
                                        (self.w[:, self.time].prod() - (1 + self.rho / 2) * self.w[:, self.time].sum() + 1 + self.rho))
            

    def step_choice_kernel(self, choice):
        # Choice vector
        choice_vector = np.zeros([self.K])
        choice_vector[choice] = 1

        # Update choice kernel (see Model 5 of Wilson and Collins, 2019)
        # Note that if chocie_step_size = 1, degenerates to Bari 2019 (choice kernel = the last choice only)
        self.choice_kernel[:, self.time] = self.choice_kernel[:, self.time - 1] \
            + self.choice_step_size * \
            (choice_vector - self.choice_kernel[:, self.time - 1])

    def act(self):  # Compatible with either fitting mode (predictive) or not (generative). It's much clear now!!

        # -- Predefined --
        # Foragers that have the pattern {AmBn} (not for fitting)
        if self.forager in ['IdealpHatGreedy']:
            return self.choice_history[0, self.time]  # Already initialized

        # Probability matching of base probabilities p (not for fitting)
        if self.forager == 'pMatching':
            choice = choose_ps(self.p_reward[:, self.time])
            self.choice_history[0, self.time] = choice
            return choice

        if self.forager == 'Random':
            return self.act_random()

        if self.forager == 'LossCounting':
            return self.act_LossCounting()

        if 'epsi' in self.forager: # 'RW1972_epsi', 'LNP_epsi'
            return self.act_EpsiGreedy()

        if self.forager in ['RW1972_softmax', 'LNP_softmax', 'Bari2019', 'Hattori2019',
                            'RW1972_softmax_CK', 'LNP_softmax_CK', 'Bari2019_CK', 'Hattori2019_CK',
                            'CANN', 'Synaptic', 'Synaptic_W>0']:   # Probabilistic (Could have choice kernel)
            return self.act_Probabilistic()

        print('No action found!!')

    def step(self, choice):  # Compatible with either fitting mode (predictive) or not (generative). It's much clear now!!

        if self.if_fit_mode:
            #  In fitting mode, retrieve choice and reward from the targeted fit_c and fit_r
            choice = self.fit_choice_history[0, self.time]  # Override choice
            # Override reward
            reward = self.fit_reward_history[choice, self.time]

        else:
            #  In generative mode, generate reward and make the state transition
            reward = self.reward_available[choice, self.time]
            # Note that according to Sutton & Barto's convention,
            self.reward_history[choice, self.time] = reward
            # this update should belong to time t+1, but here I use t for simplicity.

            # An intermediate reward status. Note the .copy()!
            reward_available_after_choice = self.reward_available[:, self.time].copy(
            )
            # The reward is depleted at the chosen lick port.
            reward_available_after_choice[choice] = 0

        # =================================================
        self.time += 1   # Time ticks here !!!
        # Doesn't terminate here to finish the final update after the last trial
        # if self.time == self.n_trials:
        #     return   # Session terminates
        # =================================================

        # Prepare reward for the next trial (if sesson did not end)
        if not self.if_fit_mode:
            # Generate the next reward status, the "or" statement ensures the baiting property, gated by self.if_baited.
            self.reward_available[:, self.time] = np.logical_or(reward_available_after_choice * self.if_baited,
                                                                np.random.uniform(0, 1, self.K) < self.p_reward[:, self.time]).astype(int)

        # Update value function etc.
        if self.forager in ['LossCounting']:
            self.step_LossCounting(reward)

        elif self.forager in ['RW1972_softmax', 'RW1972_epsi', 'Bari2019', 'Hattori2019',
                              'RW1972_softmax_CK', 'Bari2019_CK', 'Hattori2019_CK']:
            self.step_RWlike(choice, reward)

        elif self.forager in ['CANN']:
            self.step_CANN(choice, reward)
        
        elif self.forager in ['Synaptic']:
            self.step_synaptic(choice, reward)

        elif 'LNP' in self.forager: # 'LNP_softmax', 'LNP_epsilon', 'LNP_softmax_CK'
            if self.if_fit_mode:
                # Targeted history till now
                valid_reward_history = self.fit_reward_history[:, :self.time]
            else:
                # Models' history till now
                valid_reward_history = self.reward_history[:, :self.time]

            self.step_LNP(valid_reward_history)

        if '_CK' in self.forager:  # Could be independent of other foragers, so use "if" rather than "elif"
            self.step_choice_kernel(choice)

    def simulate(self):

        # =============================================================================
        # Simulate one session
        # =============================================================================
        self.reset()

        for t in range(self.n_trials):
            action = self.act()
            self.step(action)

        if self.if_fit_mode:
            # Allow the final update of action prob after the last trial (for comparing with ephys)
            action = self.act()


    def compute_foraging_eff(self, para_optim):
        # -- 1. Foraging efficiency = Sum of actual rewards / Maximum number of rewards that could have been collected --
        self.actual_rewards = np.sum(self.reward_history)
        
        '''Don't know which one is the fairest''' #???
        # Method 1: Average of max(p_reward) 
        # self.maximum_rewards = np.sum(np.max(self.p_reward, axis = 0)) 
        # Method 2: Average of sum(p_reward).   [Corrado et al 2005: efficienty = 50% for choosing only one color]
        # self.maximum_rewards = np.sum(np.sum(self.p_reward, axis = 0)) 
        # Method 3: Maximum reward given the actual reward_available (one choice per trial constraint)
        # self.maximum_rewards = np.sum(np.any(self.reward_available, axis = 0))  # Equivalent to sum(max())
        # Method 4: Sum of all ever-baited rewards (not fair)  
        # self.maximum_rewards = np.sum(np.sum(self.reward_available, axis = 0))
        
        ''' Use ideal-p^-optimal'''
        # self.maximum_rewards = self.rewards_IdealpHatGreedy
        # if not para_optim: 
        #     self.maximum_rewards = self.rewards_IdealpHatOptimal
        # else:  # If in optimization, fast and good
        self.maximum_rewards = self.rewards_IdealpHatGreedy
            
        self.foraging_efficiency = self.actual_rewards / self.maximum_rewards
        


class BanditModelRestless(BanditModel):
    
    def __init__(self, p_min=0.01, p_max=1, sigma=0.15, mean=0, **kwargs):
        super().__init__(**kwargs)
        
        self.task = 'Bandit_restless'
        self.p_min = p_min
        self.p_max = p_max
        self.sigma = sigma
        self.mean = mean
        
        self.if_baited = False
        

    def generate_p_reward(self):

        restless_bandit = RandomWalkReward(p_min=self.p_min, p_max=self.p_max, sigma=self.sigma, mean=self.mean)

        # If para_optim, fix the random seed to ensure that p_reward schedule is fixed for all candidate parameters
        # However, we should make it random during a session (see the last line of this function)
        if self.p_reward_seed_override != '':
            np.random.seed(self.p_reward_seed_override)

        while restless_bandit.trial_now < self.n_trials:     
            restless_bandit.next_trial()

        p_reward = np.vstack([restless_bandit.trial_rwd_prob['L'],
                              restless_bandit.trial_rwd_prob['R']])

        self.n_blocks = 0
        self.p_reward = p_reward
        self.block_size = []
        self.p_reward_fraction = p_reward[RIGHT, :] / \
            (np.sum(p_reward, axis=0))   # For future use
        self.p_reward_ratio = p_reward[RIGHT, :] / \
            p_reward[LEFT, :]   # For future use

        # We should make it random afterwards
        np.random.seed()
        
        self.rewards_IdealpHatOptimal = 1
        self.rewards_IdealpHatGreedy = 1
        

    def compute_foraging_eff(self, para_optim):
        
        # -- 1. Foraging efficiency = Sum of actual rewards / Maximum number of rewards that could have been collected --
        self.actual_rewards = np.sum(self.reward_history)
        
        '''Don't know which one is the fairest''' #???
        # Method 1: Average of max(p_reward) 
        self.maximum_rewards = np.sum(np.max(self.p_reward, axis = 0))
        
        # Method 2: Average of sum(p_reward).   [Corrado et al 2005: efficienty = 50% for choosing only one color]
        # self.maximum_rewards = np.sum(np.sum(self.p_reward, axis = 0)) 
        # Method 3: Maximum reward given the actual reward_available (one choice per trial constraint)
        # self.maximum_rewards = np.sum(np.any(self.reward_available, axis = 0))  # Equivalent to sum(max())
        # Method 4: Sum of all ever-baited rewards (not fair)  
        # self.maximum_rewards = np.sum(np.sum(self.reward_available, axis = 0))
            
        self.foraging_efficiency = self.actual_rewards / self.maximum_rewards




def fit_bandit(forager, fit_names, fit_bounds, choice_history, reward_history, session_num = None, 
               if_predictive = False, if_generative = False,  # Whether compute predictive or generative choice sequence
               if_history = False, fit_method = 'DE', DE_pop_size = 16, n_x0s = 1, pool = ''):
    '''
    Main fitting func and compute BIC etc.
    '''
    if if_history: 
        global fit_history
        fit_history = []
        fit_histories = []  # All histories for different initializations
        
    # === Fitting ===
    
    if fit_method == 'DE':
        
        # Use DE's own parallel method
        fitting_result = optimize.differential_evolution(func = negLL_func, args = (forager, fit_names, choice_history, reward_history, session_num, {}, []),
                                                         bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), 
                                                         mutation=(0.5, 1), recombination = 0.7, popsize = DE_pop_size, strategy = 'best1bin', 
                                                         disp = False, 
                                                         workers = 1 if pool == '' else int(mp.cpu_count()),   # For DE, use pool to control if_parallel, although we don't use pool for DE
                                                         updating = 'immediate' if pool == '' else 'deferred',
                                                         callback = callback_history if if_history else None,)
        if if_history:
            fit_history.append(fitting_result.x.copy())  # Add the final result
            fit_histories = [fit_history]  # Backward compatibility
        
    elif fit_method in ['L-BFGS-B', 'SLSQP', 'TNC', 'trust-constr']:
        
        # Do parallel initialization
        fitting_parallel_results = []
        
        if pool != '':  # Go parallel
            pool_results = []
            
            # Must use two separate for loops, one for assigning and one for harvesting!
            for nn in range(n_x0s):
                # Assign jobs
                pool_results.append(pool.apply_async(fit_each_init, args = (forager, fit_names, fit_bounds, choice_history, reward_history, session_num, fit_method, 
                                                                            None)))   # We can have multiple histories only in serial mode
            for rr in pool_results:
                # Get data    
                fitting_parallel_results.append(rr.get())
        else:
            # Serial
            for nn in range(n_x0s):
                # We can have multiple histories only in serial mode
                if if_history: fit_history = []  # Clear this history
                
                result = fit_each_init(forager, fit_names, fit_bounds, choice_history, reward_history, session_num, fit_method,
                                       callback = callback_history if if_history else None)
                
                fitting_parallel_results.append(result)
                if if_history: 
                    fit_history.append(result.x.copy())  # Add the final result
                    fit_histories.append(fit_history)
            
        # Find the global optimal
        cost = np.zeros(n_x0s)
        for nn,rr in enumerate(fitting_parallel_results):
            cost[nn] = rr.fun
        
        best_ind = np.argmin(cost)
        
        fitting_result = fitting_parallel_results[best_ind]
        if if_history and fit_histories != []:
            fit_histories.insert(0,fit_histories.pop(best_ind))  # Move the best one to the first
        
    if if_history:
        fitting_result.fit_histories = fit_histories
        
    # === For Model Comparison ===
    fitting_result.k_model = np.sum(np.diff(np.array(fit_bounds),axis=0)>0)  # Get the number of fitted parameters with non-zero range of bounds
    fitting_result.n_trials = np.shape(choice_history)[1]
    fitting_result.log_likelihood = - fitting_result.fun
    
    fitting_result.AIC = -2 * fitting_result.log_likelihood + 2 * fitting_result.k_model
    fitting_result.BIC = -2 * fitting_result.log_likelihood + fitting_result.k_model * np.log(fitting_result.n_trials)
    
    # Likelihood-Per-Trial. See Wilson 2019 (but their formula was wrong...)
    fitting_result.LPT = np.exp(fitting_result.log_likelihood / fitting_result.n_trials)  # Raw LPT without penality
    fitting_result.LPT_AIC = np.exp(- fitting_result.AIC / 2 / fitting_result.n_trials)
    fitting_result.LPT_BIC = np.exp(- fitting_result.BIC / 2 / fitting_result.n_trials)
    
    # === Rerun predictive choice sequence ===
    if if_predictive:
        
        kwargs_all = {}
        for (nn, vv) in zip(fit_names, fitting_result.x):  # Use the fitted data
            kwargs_all = {**kwargs_all, nn:vv}
        
        # Handle data from different sessions
        if session_num is None:
            session_num = np.zeros_like(choice_history)[0]  # Regard as one session
        
        unique_session = np.unique(session_num)
        predictive_choice_prob = []
        fitting_result.trial_numbers = []
        
        # -- For each session --
        for ss in unique_session:
            # Data in this session
            choice_this = choice_history[:, session_num == ss]
            reward_this = reward_history[:, session_num == ss]
            fitting_result.trial_numbers.append(np.sum(session_num == ss))
            
            # Run **PREDICTIVE** simulation    
            bandit = BanditModel(forager = forager, **kwargs_all, fit_choice_history = choice_this, fit_reward_history = reward_this)  # Into the fitting mode
            bandit.simulate()
            predictive_choice_prob.append(bandit.predictive_choice_prob)
        
        fitting_result.predictive_choice_prob = np.hstack(predictive_choice_prob)
        
    # === Run generative choice sequence ==  #!!!
        

    return fitting_result
            

import scipy.optimize as optimize
import multiprocessing as mp
# from tqdm import tqdm  # For progress bar. HH
global fit_history

    
import random

def cross_validate_bandit(forager, fit_names, fit_bounds, choice_history, reward_history, session_num = None, k_fold = 2, 
                          DE_pop_size = 16, pool = '', if_verbose = True):
    '''
    k-fold cross-validation
    '''
    
    # Split the data into k_fold parts
    n_trials = np.shape(choice_history)[1]
    trial_numbers_shuffled = np.arange(n_trials)
    random.shuffle(trial_numbers_shuffled)
    
    prediction_accuracy_test = []
    prediction_accuracy_fit = []
    prediction_accuracy_test_bias_only = []
    
    for kk in range(k_fold):
        
        test_begin = int(kk * np.floor(n_trials/k_fold))
        test_end = int((n_trials) if (kk == k_fold - 1) else (kk+1) * np.floor(n_trials/k_fold))
        test_set_this = trial_numbers_shuffled[test_begin:test_end]
        fit_set_this = np.hstack((trial_numbers_shuffled[:test_begin], trial_numbers_shuffled[test_end:]))
        
        # == Fit data using fit_set_this ==
        if if_verbose: print('%g/%g...'%(kk+1, k_fold), end = '')
        fitting_result = optimize.differential_evolution(func = negLL_func, args = (forager, fit_names, choice_history, reward_history, session_num, {}, fit_set_this),
                                                         bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), 
                                                         mutation=(0.5, 1), recombination = 0.7, popsize = DE_pop_size, strategy = 'best1bin', 
                                                         disp = False, 
                                                         workers = 1 if pool == '' else int(mp.cpu_count()),   # For DE, use pool to control if_parallel, although we don't use pool for DE
                                                         updating = 'immediate' if pool == '' else 'deferred',
                                                         callback = None,)
            
        # == Rerun predictive choice sequence and get the prediction accuracy of the test_set_this ==
        kwargs_all = {}
        for (nn, vv) in zip(fit_names, fitting_result.x):  # Use the fitted data
            kwargs_all = {**kwargs_all, nn:vv}
        
        # Handle data from different sessions
        if session_num is None:
            session_num = np.zeros_like(choice_history)[0]  # Regard as one session
        
        unique_session = np.unique(session_num)
        predictive_choice_prob = []
        fitting_result.trial_numbers = []
        
        # -- For each session --
        for ss in unique_session:
            # Data in this session
            choice_this = choice_history[:, session_num == ss]
            reward_this = reward_history[:, session_num == ss]
            
            # Run PREDICTIVE simulation    
            bandit = BanditModel(forager = forager, **kwargs_all, fit_choice_history = choice_this, fit_reward_history = reward_this)  # Into the fitting mode
            bandit.simulate()
            predictive_choice_prob.extend(bandit.predictive_choice_prob)
            
        # Get prediction accuracy of the test_set and fitting_set
        predictive_choice_prob = np.array(predictive_choice_prob)
        predictive_choice = np.argmax(predictive_choice_prob, axis = 0)
        prediction_correct = predictive_choice == choice_history[0]
        
        # Also return cross-validated prediction_accuracy_bias (Maybe this is why Hattori's bias_only is low? -- Not exactly...)
        if 'biasL' in kwargs_all:
            bias_this = kwargs_all['biasL']
            prediction_correct_bias_only = int(bias_this <= 0) == choice_history[0] # If bias_this < 0, bias predicts all rightward choices
        
        prediction_accuracy_test.append(sum(prediction_correct[test_set_this]) / len(test_set_this))
        prediction_accuracy_fit.append(sum(prediction_correct[fit_set_this]) / len(fit_set_this))
        
        prediction_accuracy_test_bias_only.append(sum(prediction_correct_bias_only[test_set_this]) / len(test_set_this))

    return prediction_accuracy_test, prediction_accuracy_fit, prediction_accuracy_test_bias_only
            



def fit_bandit(forager, fit_names, fit_bounds, choice_history, reward_history, session_num = None, 
               if_predictive = False, if_generative = False,  # Whether compute predictive or generative choice sequence
               if_history = False, fit_method = 'DE', DE_pop_size = 16, n_x0s = 1, pool = ''):
    '''
    Main fitting func and compute BIC etc.
    '''
    if if_history: 
        global fit_history
        fit_history = []
        fit_histories = []  # All histories for different initializations
        
    # === Fitting ===
    
    if fit_method == 'DE':
        
        # Use DE's own parallel method
        fitting_result = optimize.differential_evolution(func = negLL_func, args = (forager, fit_names, choice_history, reward_history, session_num, {}, []),
                                                         bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), 
                                                         mutation=(0.5, 1), recombination = 0.7, popsize = DE_pop_size, strategy = 'best1bin', 
                                                         disp = False, 
                                                         workers = 1 if pool == '' else int(mp.cpu_count()),   # For DE, use pool to control if_parallel, although we don't use pool for DE
                                                         updating = 'immediate' if pool == '' else 'deferred',
                                                         callback = callback_history if if_history else None,)
        if if_history:
            fit_history.append(fitting_result.x.copy())  # Add the final result
            fit_histories = [fit_history]  # Backward compatibility
        
    elif fit_method in ['L-BFGS-B', 'SLSQP', 'TNC', 'trust-constr']:
        
        # Do parallel initialization
        fitting_parallel_results = []
        
        if pool != '':  # Go parallel
            pool_results = []
            
            # Must use two separate for loops, one for assigning and one for harvesting!
            for nn in range(n_x0s):
                # Assign jobs
                pool_results.append(pool.apply_async(fit_each_init, args = (forager, fit_names, fit_bounds, choice_history, reward_history, session_num, fit_method, 
                                                                            None)))   # We can have multiple histories only in serial mode
            for rr in pool_results:
                # Get data    
                fitting_parallel_results.append(rr.get())
        else:
            # Serial
            for nn in range(n_x0s):
                # We can have multiple histories only in serial mode
                if if_history: fit_history = []  # Clear this history
                
                result = fit_each_init(forager, fit_names, fit_bounds, choice_history, reward_history, session_num, fit_method,
                                       callback = callback_history if if_history else None)
                
                fitting_parallel_results.append(result)
                if if_history: 
                    fit_history.append(result.x.copy())  # Add the final result
                    fit_histories.append(fit_history)
            
        # Find the global optimal
        cost = np.zeros(n_x0s)
        for nn,rr in enumerate(fitting_parallel_results):
            cost[nn] = rr.fun
        
        best_ind = np.argmin(cost)
        
        fitting_result = fitting_parallel_results[best_ind]
        if if_history and fit_histories != []:
            fit_histories.insert(0,fit_histories.pop(best_ind))  # Move the best one to the first
        
    if if_history:
        fitting_result.fit_histories = fit_histories
        
    # === For Model Comparison ===
    fitting_result.k_model = np.sum(np.diff(np.array(fit_bounds),axis=0)>0)  # Get the number of fitted parameters with non-zero range of bounds
    fitting_result.n_trials = np.shape(choice_history)[1]
    fitting_result.log_likelihood = - fitting_result.fun
    
    fitting_result.AIC = -2 * fitting_result.log_likelihood + 2 * fitting_result.k_model
    fitting_result.BIC = -2 * fitting_result.log_likelihood + fitting_result.k_model * np.log(fitting_result.n_trials)
    
    # Likelihood-Per-Trial. See Wilson 2019 (but their formula was wrong...)
    fitting_result.LPT = np.exp(fitting_result.log_likelihood / fitting_result.n_trials)  # Raw LPT without penality
    fitting_result.LPT_AIC = np.exp(- fitting_result.AIC / 2 / fitting_result.n_trials)
    fitting_result.LPT_BIC = np.exp(- fitting_result.BIC / 2 / fitting_result.n_trials)
    
    # === Rerun predictive choice sequence ===
    if if_predictive:
        
        kwargs_all = {}
        for (nn, vv) in zip(fit_names, fitting_result.x):  # Use the fitted data
            kwargs_all = {**kwargs_all, nn:vv}
        
        # Handle data from different sessions
        if session_num is None:
            session_num = np.zeros_like(choice_history)[0]  # Regard as one session
        
        unique_session = np.unique(session_num)
        predictive_choice_prob = []
        fitting_result.trial_numbers = []
        
        # -- For each session --
        for ss in unique_session:
            # Data in this session
            choice_this = choice_history[:, session_num == ss]
            reward_this = reward_history[:, session_num == ss]
            fitting_result.trial_numbers.append(np.sum(session_num == ss))
            
            # Run **PREDICTIVE** simulation    
            bandit = BanditModel(forager = forager, **kwargs_all, fit_choice_history = choice_this, fit_reward_history = reward_this)  # Into the fitting mode
            bandit.simulate()
            predictive_choice_prob.append(bandit.predictive_choice_prob)
        
        fitting_result.predictive_choice_prob = np.hstack(predictive_choice_prob)
        
    # === Run generative choice sequence ==  #!!!
        

    return fitting_result
            

import random

def cross_validate_bandit(forager, fit_names, fit_bounds, choice_history, reward_history, session_num = None, k_fold = 2, 
                          DE_pop_size = 16, pool = '', if_verbose = True):
    '''
    k-fold cross-validation
    '''
    
    # Split the data into k_fold parts
    n_trials = np.shape(choice_history)[1]
    trial_numbers_shuffled = np.arange(n_trials)
    random.shuffle(trial_numbers_shuffled)
    
    prediction_accuracy_test = []
    prediction_accuracy_fit = []
    prediction_accuracy_test_bias_only = []
    
    for kk in range(k_fold):
        
        test_begin = int(kk * np.floor(n_trials/k_fold))
        test_end = int((n_trials) if (kk == k_fold - 1) else (kk+1) * np.floor(n_trials/k_fold))
        test_set_this = trial_numbers_shuffled[test_begin:test_end]
        fit_set_this = np.hstack((trial_numbers_shuffled[:test_begin], trial_numbers_shuffled[test_end:]))
        
        # == Fit data using fit_set_this ==
        if if_verbose: print('%g/%g...'%(kk+1, k_fold), end = '')
        fitting_result = optimize.differential_evolution(func = negLL_func, args = (forager, fit_names, choice_history, reward_history, session_num, {}, fit_set_this),
                                                         bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), 
                                                         mutation=(0.5, 1), recombination = 0.7, popsize = DE_pop_size, strategy = 'best1bin', 
                                                         disp = False, 
                                                         workers = 1 if pool == '' else int(mp.cpu_count()),   # For DE, use pool to control if_parallel, although we don't use pool for DE
                                                         updating = 'immediate' if pool == '' else 'deferred',
                                                         callback = None,)
            
        # == Rerun predictive choice sequence and get the prediction accuracy of the test_set_this ==
        kwargs_all = {}
        for (nn, vv) in zip(fit_names, fitting_result.x):  # Use the fitted data
            kwargs_all = {**kwargs_all, nn:vv}
        
        # Handle data from different sessions
        if session_num is None:
            session_num = np.zeros_like(choice_history)[0]  # Regard as one session
        
        unique_session = np.unique(session_num)
        predictive_choice_prob = []
        fitting_result.trial_numbers = []
        
        # -- For each session --
        for ss in unique_session:
            # Data in this session
            choice_this = choice_history[:, session_num == ss]
            reward_this = reward_history[:, session_num == ss]
            
            # Run PREDICTIVE simulation    
            bandit = BanditModel(forager = forager, **kwargs_all, fit_choice_history = choice_this, fit_reward_history = reward_this)  # Into the fitting mode
            bandit.simulate()
            predictive_choice_prob.extend(bandit.predictive_choice_prob)
            
        # Get prediction accuracy of the test_set and fitting_set
        predictive_choice_prob = np.array(predictive_choice_prob)
        predictive_choice = np.argmax(predictive_choice_prob, axis = 0)
        prediction_correct = predictive_choice == choice_history[0]
        
        # Also return cross-validated prediction_accuracy_bias (Maybe this is why Hattori's bias_only is low? -- Not exactly...)
        if 'biasL' in kwargs_all:
            bias_this = kwargs_all['biasL']
            prediction_correct_bias_only = int(bias_this <= 0) == choice_history[0] # If bias_this < 0, bias predicts all rightward choices
        
        prediction_accuracy_test.append(sum(prediction_correct[test_set_this]) / len(test_set_this))
        prediction_accuracy_fit.append(sum(prediction_correct[fit_set_this]) / len(fit_set_this))
        
        prediction_accuracy_test_bias_only.append(sum(prediction_correct_bias_only[test_set_this]) / len(test_set_this))

    return prediction_accuracy_test, prediction_accuracy_fit, prediction_accuracy_test_bias_only
            



def moving_average(a, n=3) :
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd

from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm

# import models.bandit_model_comparison
# import matplotlib as mpl 
# mpl.rcParams['figure.dpi'] = 300

# matplotlib.use('qt5agg')
plt.rcParams.update({'font.size': 14, 'figure.dpi': 150})

# from utils.helper_func import seaborn_style
# seaborn_style()

def plot_para_recovery(forager, true_paras, fitted_paras, para_names, para_bounds, para_scales, para_color_code, para_2ds, n_trials, fit_method):
    # sns.reset_orig()
    n_paras, n_models = np.shape(fitted_paras)
    n_para_2ds = len(para_2ds)
    if para_scales is None: 
        para_scales = ['linear'] * n_paras
        
    # Color coded: 1 or the noise level (epsilon or softmax_temperature) 
    if para_color_code is None:
        if 'epsilon' in para_names:
            para_color_code = para_names.index('epsilon')
        elif 'softmax_temperature' in para_names:
            para_color_code = para_names.index('softmax_temperature')
        else:
            para_color_code = 1
            
    nn = min(4, n_paras + n_para_2ds)  # Column number
    mm = np.ceil((n_paras + n_para_2ds)/nn).astype(int)
    fig = plt.figure(figsize=(nn*4, mm*5), dpi = 100)
    
    fig.text(0.05,0.90,'Parameter Recovery: %s, Method: %s, N_trials = %g, N_runs = %g\nColor code: %s' % (forager, fit_method, n_trials, n_models, para_names[para_color_code]), fontsize = 15)

    gs = GridSpec(mm, nn, wspace=0.4, hspace=0.3, bottom=0.15, top=0.80, left=0.07, right=0.97) 
    
    xmin = np.min(true_paras[para_color_code,:])
    xmax = np.max(true_paras[para_color_code,:])
    if para_scales[para_color_code] == 'log':
        xmin = np.min(true_paras[para_color_code,:])
        xmax = np.max(true_paras[para_color_code,:])
        colors = cm.copper((np.log(true_paras[para_color_code,:])-np.log(xmin))/(np.log(xmax)-np.log(xmin)+1e-6)) # Use second as color (backward compatibility)
    else:
        colors = cm.copper((true_paras[para_color_code,:]-xmin)/(xmax-xmin+1e-6)) # Use second as color (backward compatibility)
      
    
    # 1. 1-D plot
    for pp in range(n_paras):
        ax = fig.add_subplot(gs[np.floor(pp/nn).astype(int), np.mod(pp,nn).astype(int)])
        plt.plot([para_bounds[0][pp], para_bounds[1][pp]], [para_bounds[0][pp], para_bounds[1][pp]],'k--',linewidth=1)
        
        # Raw data
        plt.scatter(true_paras[pp,:], fitted_paras[pp,:], marker = 'o', facecolors='none', s = 100, c = colors, alpha=0.7)
        
        if n_models > 1:   # Linear regression
            if para_scales[pp] == 'linear':
                x = true_paras[pp,:]
                y = fitted_paras[pp,:]
            else:  #  Use log10 if needed
                x = np.log10(true_paras[pp,:])
                y = np.log10(fitted_paras[pp,:])
                
            model = sm.OLS(y, sm.add_constant(x)).fit()
            b, k = model.params  
            r_square, p = (model.rsquared, model.pvalues)
            
            if para_scales[pp] == 'linear':
                plt.plot([min(x),max(x)], [k*min(x)+b, k*max(x)+b,], '-k', label = 'r^2 = %.2g\np = %.2g'%(r_square, p[1]))
            else:  #  Use log10 if needed
                plt.plot([10**min(x), 10**max(x)], [10**(k*min(x)+b), 10**(k*max(x)+b),], '-k', label = 'r^2 = %.2g\np = %.2g'%(r_square, p[1]))
        
        ax.set_xscale(para_scales[pp])
        ax.set_yscale(para_scales[pp])
        ax.legend()
        
        plt.title(para_names[pp])
        plt.xlabel('True')
        plt.ylabel('Fitted')
        plt.axis('square')
        
    # 2. 2-D plot
    
    for pp, para_2d in enumerate(para_2ds):
        
        ax = fig.add_subplot(gs[np.floor((pp+n_paras)/nn).astype(int), np.mod(pp+n_paras,nn).astype(int)])    
        legend_plotted = False
        
        # Connected true and fitted data
        for n in range(n_models):
            plt.plot(true_paras[para_2d[0],n], true_paras[para_2d[1],n],'ok', markersize=11, fillstyle='none', c = colors[n], label = 'True' if not legend_plotted else '',alpha=.7)
            plt.plot(fitted_paras[para_2d[0],n], fitted_paras[para_2d[1],n],'ok', markersize=7, c = colors[n], label = 'Fitted' if not legend_plotted else '',alpha=.7)
            legend_plotted = True
            
            plt.plot([true_paras[para_2d[0],n], fitted_paras[para_2d[0],n]], [true_paras[para_2d[1],n], fitted_paras[para_2d[1],n]],'-', linewidth=1, c = colors[n])
            
        # Draw the fitting bounds
        x1 = para_bounds[0][para_2d[0]]
        y1 = para_bounds[0][para_2d[1]]
        x2 = para_bounds[1][para_2d[0]]
        y2 = para_bounds[1][para_2d[1]]
        
        plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'k--',linewidth=1)
        
        plt.xlabel(para_names[para_2d[0]])
        plt.ylabel(para_names[para_2d[1]])
        
        if para_scales[para_2d[0]] == 'linear' and para_scales[para_2d[1]] == 'linear':
            ax.set_aspect(1.0/ax.get_data_ratio())  # This is the correct way of setting square display
        
        ax.set_xscale(para_scales[para_2d[0]])
        ax.set_yscale(para_scales[para_2d[1]])

    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()


def plot_LL_surface(forager, LLsurfaces, CI_cutoff_LPTs, para_names, para_2ds, para_grids, para_scales, true_para, fitted_para, fit_history, fit_method, n_trials):
    
    sns.reset_orig()
            
    n_para_2ds = len(para_2ds)
    
    # ==== Figure setting ===
    nn_ax = min(3, n_para_2ds) # Column number
    mm_ax = np.ceil(n_para_2ds/nn_ax).astype(int)
    fig = plt.figure(figsize=(2.5+nn_ax*5, 1.5+mm_ax*5), dpi = 100)
    gs = GridSpec(mm_ax, nn_ax, wspace=0.2, hspace=0.35, bottom=0.1, top=0.84, left=0.07, right=0.97) 
    fig.text(0.05,0.88,'Likelihood Per Trial = p(data|paras, model)^(1/T): %s,\n Method: %s, N_trials = %g\n  True values: %s\nFitted values: %s' % (forager, fit_method, n_trials, 
                                                                                                                            np.round(true_para,3), np.round(fitted_para,3)),fontsize = 13)

    # ==== Plot each LL surface ===
    for ppp,(LLs, CI_cutoff_LPT, ps, para_2d) in enumerate(zip(LLsurfaces, CI_cutoff_LPTs, para_grids, para_2ds)):
    
        ax = fig.add_subplot(gs[np.floor(ppp/nn_ax).astype(int), np.mod(ppp,nn_ax).astype(int)]) 
        
        fitted_para_this = [fitted_para[para_2d[0]], fitted_para[para_2d[1]]]
        true_para_this = [true_para[para_2d[0]], true_para[para_2d[1]]]     
        para_names_this = [para_names[para_2d[0]], para_names[para_2d[1]]]
        para_scale = [para_scales[para_2d[0]], para_scales[para_2d[1]]]
                            
        for ii in range(2):
            if para_scale[ii] == 'log':
                ps[ii] = np.log10(ps[ii])
                fitted_para_this[ii] = np.log10(fitted_para_this[ii])
                true_para_this[ii] = np.log10(true_para_this[ii])
        
        dx = ps[0][1]-ps[0][0]
        dy = ps[1][1]-ps[1][0]
        extent=[ps[0].min()-dx/2, ps[0].max()+dx/2, ps[1].min()-dy/2, ps[1].max()+dy/2]

        # -- Gaussian filtering ---

        if dx > 0 and dy > 0:
            plt.imshow(LLs, cmap='plasma', extent=extent, interpolation='none', origin='lower')
            plt.colorbar()
        # plt.pcolor(pp1, pp2, LLs, cmap='RdBu', vmin=z_min, vmax=z_max)
        
        plt.contour(LLs, colors='grey', levels = 20, extent=extent, linewidths=0.7)
        # plt.contour(-np.log(-LLs), colors='grey', levels = 20, extent=extent, linewidths=0.7)
        
        # -- Cutoff LPT --
        plt.contour(LLs, levels = [CI_cutoff_LPT], colors = 'r', extent=extent)
        
        # ==== True value ==== 
        plt.plot(true_para_this[0], true_para_this[1],'ob', markersize = 20, markeredgewidth=3, fillstyle='none')
        
        # ==== Fitting history (may have many) ==== 
        if fit_history != []:
            
            # Compatible with one history (global optimizers) or multiple histories (local optimizers)
            for nn, hh in reversed(list(enumerate(fit_history))):  
                hh = np.array(hh)
                hh = hh[:,(para_2d[0], para_2d[1])]  # The user-defined 2-d subspace
                
                for ii in range(2):
                    if para_scale[ii] == 'log': hh[:,ii] = np.log10(hh[:,ii])
                
                sizes = 100 * np.linspace(0.1,1,np.shape(hh)[0])
                plt.scatter(hh[:,0], hh[:,1], s = sizes, c = 'k' if nn == 0 else None)
                plt.plot(hh[:,0], hh[:,1], '-' if nn == 0 else ':', color = 'k' if nn == 0 else None)
                
        # ==== Final fitted result ====
        plt.plot(fitted_para_this[0], fitted_para_this[1],'Xb', markersize=17)
    
        ax.set_aspect(1.0/ax.get_data_ratio()) 
        
        plt.xlabel(('log10 ' if para_scale[0] == 'log' else '') + para_names_this[0])
        plt.ylabel(('log10 ' if para_scale[1] == 'log' else '') + para_names_this[1])
        
    
    plt.show()
    
def plot_session_lightweight(fake_data, fitted_data = None, smooth_factor = 5, base_color = 'y'):
    # sns.reset_orig()
    sns.set(style="ticks", context="paper", font_scale=1.4)

    choice_history, reward_history, p_reward = fake_data
    
    # == Fetch data ==
    n_trials = np.shape(choice_history)[1]
    
    p_reward_fraction = p_reward[1,:] / (np.sum(p_reward, axis = 0))
                                      
    rewarded_trials = np.any(reward_history, axis = 0)
    unrewarded_trials = np.logical_not(rewarded_trials)
    
    # == Choice trace ==
    fig = plt.figure(figsize=(9, 4), dpi = 150)
        
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left = 0.1, right=0.8)

    # Rewarded trials
    ax.plot(np.nonzero(rewarded_trials)[0], 0.5 + (choice_history[0,rewarded_trials]-0.5) * 1.4, 
            'k|',color='black',markersize=20, markeredgewidth=2)

    # Unrewarded trials
    ax.plot(np.nonzero(unrewarded_trials)[0], 0.5 + (choice_history[0,unrewarded_trials] - 0.5) * 1.4, 
            '|',color='gray', markersize=10, markeredgewidth=1)
    
    # Base probability
    ax.plot(np.arange(0, n_trials), p_reward_fraction, color=base_color, label = 'base rew. prob.', lw = 1.5)
    
    # Smoothed choice history
    y = moving_average(choice_history, smooth_factor)
    x = np.arange(0, len(y)) + int(smooth_factor/2)


    print('the shape of x is ', x.shape)
    print('the shape of y is ', y.shape)
    ax.plot(x, y, linewidth = 1.5, color='black', label = 'choice (smooth = %g)' % smooth_factor)
    
    # For each session, if any
    if fitted_data is not None:
        ax.plot(np.arange(0, n_trials+1), fitted_data[1,:], linewidth = 1.5, label = 'model') 

    ax.legend(fontsize = 10, loc=1, bbox_to_anchor=(0.985, 0.89), bbox_transform=plt.gcf().transFigure)
     
    ax.set_yticks([0,1])
    ax.set_yticklabels(['Left','Right'])
    # ax.set_xlim(0,300)
    
    # fig.tight_layout() 
    sns.despine(trim=True)

    return ax
    
def plot_model_comparison_predictive_choice_prob(model_comparison, smooth_factor = 5):
    # sns.reset_orig()
    
    choice_history, reward_history, p_reward, trial_numbers = model_comparison.fit_choice_history, model_comparison.fit_reward_history, model_comparison.p_reward, model_comparison.trial_numbers
    if not hasattr(model_comparison,'plot_predictive'):
        model_comparison.plot_predictive = [1,2,3]
        
    n_trials = np.shape(choice_history)[1]

    ax = plot_session_lightweight([choice_history, reward_history, p_reward], smooth_factor = smooth_factor)
    # Predictive choice prob
    for bb in model_comparison.plot_predictive:
        bb = bb - 1
        if bb < len(model_comparison.results):
            this_id = model_comparison.results_sort.index[bb] - 1
            this_choice_prob = model_comparison.results_raw[this_id].predictive_choice_prob
            this_result = model_comparison.results_sort.iloc[bb]
           
            ax.plot(np.arange(0, n_trials+1), this_choice_prob[1,:] , linewidth = max(1.5-0.3*bb,0.2), 
                    label = 'Model %g: %s, Km = %g\n%s\n%s' % (bb+1, this_result.model, this_result.Km, 
                                                                                        this_result.para_notation, this_result.para_fitted))
    
    # Plot session starts
    if len(trial_numbers) > 1:  # More than one sessions
        for session_start in np.cumsum([0, *trial_numbers[:-1]]):
            plt.axvline(session_start, color='b', linestyle='--', linewidth = 2)
            try:
                plt.text(session_start + 1, 1, '%g' % model_comparison.session_num[session_start], fontsize = 10, color='b')
            except:
                pass

    ax.legend(fontsize = 7, loc=1, bbox_to_anchor=(0.985, 0.89), bbox_transform=plt.gcf().transFigure)
     
    # ax.set_xlim(0,300)
    
    # fig.tight_layout() 
    sns.set()
    return

def plot_model_comparison_result(model_comparison):
    sns.set()
    
    results = model_comparison.results
    
    # Update notations
    para_notation_with_best_fit = []
    for i, row in results.iterrows():
        para_notation_with_best_fit.append('('+str(i)+') '+row.para_notation + '\n' + str(np.round(row.para_fitted,2)))
        
    results['para_notation_with_best_fit'] = para_notation_with_best_fit
        
    fig = plt.figure(figsize=(12, 8), dpi = 150)
    gs = GridSpec(1, 5, wspace = 0.1, bottom = 0.1, top = 0.85, left = 0.23, right = 0.95)
    
    
    # -- 1. LPT -- 
    ax = fig.add_subplot(gs[0, 0])
    s = sns.barplot(x = 'LPT', y = 'para_notation_with_best_fit', data = results, color = 'grey')
    s.set_xlim(min(0.5,np.min(np.min(model_comparison.results[['LPT_AIC','LPT_BIC']]))) - 0.005)
    plt.axvline(0.5, color='k', linestyle='--')
    s.set_ylabel('')
    s.set_xlabel('Likelihood per trial')

    # -- 2. AIC, BIC raw --
    ax = fig.add_subplot(gs[0, 1])
    df = pd.melt(results[['para_notation_with_best_fit','AIC','BIC']], 
                 id_vars = 'para_notation_with_best_fit', var_name = '', value_name= 'IC')
    s = sns.barplot(x = 'IC', y = 'para_notation_with_best_fit', hue = '', data = df)

    # Annotation
    x_max = max(plt.xlim())
    ylim = plt.ylim()
    best_AIC = np.where(results.best_model_AIC)[0][0]
    plt.plot(x_max, best_AIC - 0.2, '*', markersize = 15)
    best_BIC = np.where(results.best_model_BIC)[0][0]
    plt.plot(x_max, best_BIC + 0.2, '*', markersize = 15)
    plt.ylim(ylim)
    
    s.set_yticklabels('')
    s.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', ncol = 1)
    s.set_ylabel('')
    s.set_xlabel('AIC or BIC')

    # -- 3. log10_BayesFactor --
    ax = fig.add_subplot(gs[0, 2])
    df = pd.melt(results[['para_notation_with_best_fit','log10_BF_AIC','log10_BF_BIC']], 
                 id_vars = 'para_notation_with_best_fit', var_name = '', value_name= 'log10 (Bayes factor)')
    s = sns.barplot(x = 'log10 (Bayes factor)', y = 'para_notation_with_best_fit', hue = '', data = df)
    h_d = plt.axvline(-2, color='r', linestyle='--', label = 'decisive')
    s.legend(handles = [h_d,], bbox_to_anchor=(0,1.02,1,0.2), loc='lower left')
    # s.invert_xaxis()
    s.set_xlabel('log$_{10}\\frac{p(model)}{p(best\,model)}$')
    s.set_ylabel('')
    s.set_yticklabels('')
    
    # -- 4. Model weight --
    ax = fig.add_subplot(gs[0, 3])
    df = pd.melt(results[['para_notation_with_best_fit','model_weight_AIC','model_weight_BIC']], 
                 id_vars = 'para_notation_with_best_fit', var_name = '', value_name= 'Model weight')
    s = sns.barplot(x = 'Model weight', y = 'para_notation_with_best_fit', hue = '', data = df)
    ax.legend_.remove()
    plt.xlim([0,1.05])
    plt.axvline(1, color='k', linestyle='--')
    s.set_ylabel('')
    s.set_yticklabels('')
    
    # -- 5. Prediction accuracy --
    
    prediction_accuracy_NONCV = np.zeros(len(results))
    
    for rr, raw in enumerate(model_comparison.results_raw):
        this_predictive_choice_prob = raw.predictive_choice_prob
        this_predictive_choice = np.argmax(this_predictive_choice_prob, axis = 0)
        prediction_accuracy_NONCV[rr] = np.sum(this_predictive_choice[1:] == model_comparison.fit_choice_history[0,:]) / model_comparison.n_trials
        
    results['prediction_accuracy_NONCV'] = prediction_accuracy_NONCV * 100

    ax = fig.add_subplot(gs[0, 4])
    s = sns.barplot(x = 'prediction_accuracy_NONCV', y = 'para_notation_with_best_fit', data = results, color = 'grey')
    plt.axvline(50, color='k', linestyle='--')
    ax.set_xlim(min(50,np.min(np.min(model_comparison.results[['prediction_accuracy_NONCV']]))) - 5)
    ax.set_ylabel('')
    ax.set_xlabel('Prediction_accuracy_NONCV')
    s.set_yticklabels('')

    return

def plot_confusion_matrix(confusion_results, order = None):
    sns.set()
    
    n_runs = np.sum(~np.isnan(confusion_results['raw_AIC'][0,0,:]))
    n_trials = confusion_results['n_trials']
    
    # === Get notations ==
    model_notations = confusion_results['models_notations']
    
    # Reorder if needed
    if order is not None:
        model_notations = ['('+str(ii+1)+') '+model_notations[imodel] for ii,imodel in enumerate(order)]
    else:
        model_notations = ['('+str(ii+1)+') '+ mm for ii,mm in enumerate(model_notations)]
        
    # === Plotting ==
    contents = [
               [['confusion_best_model_AIC','inversion_best_model_AIC'],
               ['confusion_best_model_BIC','inversion_best_model_BIC']],
               [['confusion_log10_BF_AIC','confusion_AIC'],
               ['confusion_log10_BF_BIC','confusion_BIC']],
               ]

    for cc, content in enumerate(contents):    
        fig = plt.figure(figsize=(10, 8.5))
        fig.text(0.05,0.97,'Model Recovery: n_trials = %g, n_runs = %g, True →, Fitted ↓' % (n_trials, n_runs))

        gs = GridSpec(2, 2, wspace=0.15, hspace=0.1, bottom=0.02, top=0.8, left=0.15, right=0.97)
    
        for ii in range(2):
            for jj in range(2):
                
                # Get data
                data = confusion_results[content[ii][jj]]
                
                # Reorder
                if order is not None:
                    data = data[:, np.array(order) - 1]
                    data = data[np.array(order) - 1, :]
                    
                # -- Plot --
                ax = fig.add_subplot(gs[ii, jj])
                
                # I transpose the data here so that the columns are ground truth and the rows are fitted results,
                # which is better aligned to the model comparison plot and the model-comparison-as-a-function-of-session in the real data.
                if cc == 0:
                    sns.heatmap(data.T, annot = True, fmt=".2g", ax = ax, square = True, annot_kws={"size": 10})
                else:
                    if jj == 0:
                        sns.heatmap(data.T, annot = True, fmt=".2g", ax = ax, square = True, annot_kws={"size": 10}, vmin=-2, vmax=0)
                    else:
                        sns.heatmap(data.T, annot = False, fmt=".2g", ax = ax, square = True, annot_kws={"size": 10})
                        
                set_label(ax, ii,jj, model_notations)
                plt.title(content[ii][jj])
        
        fig.show()
        

def set_label(h,ii,jj, model_notations):
    
    if jj == 0:
        h.set_yticklabels(model_notations, rotation = 0)
    else:
        h.set_yticklabels('')
        
    if ii == 0:
        h.set_xticklabels(model_notations, rotation = 45, ha = 'left')
        h.xaxis.tick_top()
    else:
        h.set_xticklabels('')
             
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:15:49 2020

@author: Han
"""
import pandas as pd
import time

from IPython.display import display

# Default models (reordered with hindsight results). Use the format: [forager, [para_names], [lower bounds], [higher bounds]]
MODELS = [
            # No bias (1-8)
            ['LossCounting', ['loss_count_threshold_mean', 'loss_count_threshold_std'], [0,0], [40,10]],                   
            ['RW1972_epsi', ['learn_rate', 'epsilon'],[0, 0],[1, 1]],
            ['LNP_softmax',  ['tau1', 'softmax_temperature'], [1e-3, 1e-2], [100, 15]],                 
            ['LNP_softmax', ['tau1', 'tau2', 'w_tau1', 'softmax_temperature'],[1e-3, 1e-1, 0, 1e-2],[15, 40, 1, 15]],                 
            ['RW1972_softmax', ['learn_rate', 'softmax_temperature'],[0, 1e-2],[1, 15]],
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'softmax_temperature'],[0, 0, 1e-2],[1, 1, 15]],
            ['Bari2019', ['learn_rate', 'forget_rate', 'softmax_temperature'],[0, 0, 1e-2],[1, 1, 15]],
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'softmax_temperature'],[0, 0, 0, 1e-2],[1, 1, 1, 15]],
            
            # With bias (9-15)
            ['RW1972_epsi', ['learn_rate', 'epsilon', 'biasL'],[0, 0, -0.5],[1, 1, 0.5]],
            ['LNP_softmax',  ['tau1', 'softmax_temperature', 'biasL'], [1e-3, 1e-2, -5], [100, 15, 5]],                 
            ['LNP_softmax', ['tau1', 'tau2', 'w_tau1', 'softmax_temperature', 'biasL'],[1e-3, 1e-1, 0, 1e-2, -5],[15, 40, 1, 15, 5]],                 
            ['RW1972_softmax', ['learn_rate', 'softmax_temperature', 'biasL'],[0, 1e-2, -5],[1, 15, 5]],
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'softmax_temperature', 'biasL'],[0, 0, 1e-2, -5],[1, 1, 15, 5]],
            ['Bari2019', ['learn_rate', 'forget_rate', 'softmax_temperature', 'biasL'],[0, 0, 1e-2, -5],[1, 1, 15, 5]],
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'softmax_temperature', 'biasL'],[0, 0, 0, 1e-2, -5],[1, 1, 1, 15, 5]],
            
            # With bias and choice kernel (16-21)
            ['LNP_softmax_CK',  ['tau1', 'softmax_temperature', 'biasL', 'choice_step_size','choice_softmax_temperature'], 
                             [1e-3, 1e-2, -5, 0, 1e-2], [100, 15, 5, 1, 20]],                 
            ['LNP_softmax_CK', ['tau1', 'tau2', 'w_tau1', 'softmax_temperature', 'biasL', 'choice_step_size','choice_softmax_temperature'],
                             [1e-3, 1e-1, 0, 1e-2, -5, 0, 1e-2],[15, 40, 1, 15, 5, 1, 20]],                 
            ['RW1972_softmax_CK', ['learn_rate', 'softmax_temperature', 'biasL', 'choice_step_size','choice_softmax_temperature'],
                             [0, 1e-2, -5, 0, 1e-2],[1, 15, 5, 1, 20]],
            ['Hattori2019_CK', ['learn_rate_rew', 'learn_rate_unrew', 'softmax_temperature', 'biasL', 'choice_step_size','choice_softmax_temperature'],
                             [0, 0, 1e-2, -5, 0, 1e-2],[1, 1, 15, 5, 1, 20]],
            ['Bari2019_CK', ['learn_rate', 'forget_rate', 'softmax_temperature', 'biasL', 'choice_step_size','choice_softmax_temperature'],
                             [0, 0, 1e-2, -5, 0, 1e-2],[1, 1, 15, 5, 1, 20]],
            ['Hattori2019_CK', ['learn_rate_rew','learn_rate_unrew', 'forget_rate','softmax_temperature', 'biasL', 'choice_step_size','choice_softmax_temperature'],
                               [0, 0, 0, 1e-2, -5, 0, 1e-2],[1, 1, 1, 15, 5, 1, 20]],
            # ['Hattori2019_CK', ['learn_rate_rew','learn_rate_unrew', 'forget_rate','softmax_temperature', 'biasL', 'choice_step_size','choice_softmax_temperature'],
            #                    [0, 0, 0, 1e-2, -5, 1, 1e-2],[1, 1, 1, 15, 5, 1, 20]],  # choice_step_size fixed at 1 --> Bari 2019: only the last choice matters
            
         ]

# Define notations
PARA_NOTATIONS = {'loss_count_threshold_mean': '$\\mu_{LC}$',
            'loss_count_threshold_std': '$\\sigma_{LC}$',
            'tau1': '$\\tau_1$',
            'tau2': '$\\tau_2$',
            'w_tau1': '$w_{\\tau_1}$',
            'learn_rate': '$\\alpha$',   
            'learn_rate_rew': '$\\alpha_{rew}$',   
            'learn_rate_unrew': '$\\alpha_{unr}$',   
            'forget_rate': '$\\delta$',
            'softmax_temperature': '$\\sigma$',
            'epsilon': '$\\epsilon$',
            'biasL': '$b_L$',
            'biasR': '$b_R$',
            'choice_step_size': '$\\alpha_c$',
            'choice_softmax_temperature': '$\\sigma_c$',
            }


class BanditModelComparison:
    
    '''
    A new class that can define models, receive data, do fitting, and generate plots.
    This is the minimized module that can be plugged into Datajoint for real data.
    
    '''
    
    def __init__(self, choice_history, reward_history, p_reward = None, session_num = None, models = None):
        """

        Parameters
        ----------
        choice_history, reward_history, (p_reward), (session_num)
            DESCRIPTION. p_reward is only for plotting or generative validation; session_num is for pooling across sessions
        models : list of integers or models, optional
            DESCRIPTION. If it's a list of integers, the models will be selected from the pre-defined models.
            If it's a list of models, then it will be used directly. Use the format: [forager, [para_names], [lower bounds], [higher bounds]]
            The default is None (using all pre-defined models).
        Returns
        -------
        None.

        """
        
        if models is None:  
            self.models = MODELS
        elif type(models[0]) is int:
            self.models = [MODELS[i-1] for i in models]
        else:
            self.models = models
            
        self.fit_choice_history, self.fit_reward_history, self.p_reward, self.session_num = choice_history, reward_history, p_reward, session_num
        self.K, self.n_trials = np.shape(self.fit_reward_history)
        assert np.shape(self.fit_choice_history)[1] == self.n_trials, 'Choice length should be equal to reward length!'
        
        return
        
    def fit(self, fit_method = 'DE', fit_settings = {'DE_pop_size': 16}, pool = '',
                  if_verbose = True, 
                  plot_predictive = None,  # E.g.: 0,1,2,-1: The best, 2nd, 3rd and the worst model
                  plot_generative = None):
        
        self.results_raw = []
        self.results = pd.DataFrame()
        
        if if_verbose: print('=== Model Comparison ===\nMethods = %s, %s, pool = %s' % (fit_method, fit_settings, pool!=''))
        for mm, model in enumerate(self.models):
            # == Get settings for this model ==
            forager, fit_names, fit_lb, fit_ub = model
            fit_bounds = [fit_lb, fit_ub]
            
            para_notation = ''
            Km = 0
            
            for name, lb, ub in zip(fit_names, fit_lb, fit_ub):
                # == Generate notation ==
                if lb < ub:
                    para_notation += PARA_NOTATIONS[name] + ', '
                    Km += 1
            
            para_notation = para_notation[:-2]
            
            # == Do fitting here ==
            #  Km = np.sum(np.diff(np.array(fit_bounds),axis=0)>0)
            
            if if_verbose: print('Model %g/%g: %15s, Km = %g ...'%(mm+1, len(self.models), forager, Km), end='')
            start = time.time()
                
            result_this = fit_bandit(forager, fit_names, fit_bounds, self.fit_choice_history, self.fit_reward_history, self.session_num,
                                     fit_method = fit_method, **fit_settings, 
                                     pool = pool, if_predictive = True) #plot_predictive is not None)
            
            if if_verbose: print(' AIC = %g, BIC = %g (done in %.3g secs)' % (result_this.AIC, result_this.BIC, time.time()-start) )
            self.results_raw.append(result_this)
            # self.results = self.results.append(pd.DataFrame({'model': [forager], 'Km': Km, 'AIC': result_this.AIC, 'BIC': result_this.BIC, 
            #                         'LPT_AIC': result_this.LPT_AIC, 'LPT_BIC': result_this.LPT_BIC, 'LPT': result_this.LPT,
            #                         'para_names': [fit_names], 'para_bounds': [fit_bounds], 
            #                         'para_notation': [para_notation], 'para_fitted': [np.round(result_this.x,3)]}, index = [mm+1]))

            new_row = pd.DataFrame({
                'model': [forager],
                'Km': Km,
                'AIC': result_this.AIC,
                'BIC': result_this.BIC,
                'LPT_AIC': result_this.LPT_AIC,
                'LPT_BIC': result_this.LPT_BIC,
                'LPT': result_this.LPT,
                'para_names': [fit_names],
                'para_bounds': [fit_bounds],
                'para_notation': [para_notation],
                'para_fitted': [np.round(result_this.x, 3)]
            }, index=[mm + 1])

            self.results = pd.concat([self.results, new_row])

        # == Reorganize data ==
        delta_AIC = self.results.AIC - np.min(self.results.AIC) 
        delta_BIC = self.results.BIC - np.min(self.results.BIC)

        # Relative likelihood = Bayes factor = p_model/p_best = exp( - delta_AIC / 2)
        self.results['relative_likelihood_AIC'] = np.exp( - delta_AIC / 2)
        self.results['relative_likelihood_BIC'] = np.exp( - delta_BIC / 2)

        # Model weight = Relative likelihood / sum(Relative likelihood)
        self.results['model_weight_AIC'] = self.results['relative_likelihood_AIC'] / np.sum(self.results['relative_likelihood_AIC'])
        self.results['model_weight_BIC'] = self.results['relative_likelihood_BIC'] / np.sum(self.results['relative_likelihood_BIC'])
        
        # log_10 (Bayes factor) = log_10 (exp( - delta_AIC / 2)) = (-delta_AIC / 2) / log(10)
        self.results['log10_BF_AIC'] = - delta_AIC/2 / np.log(10) # Calculate log10(Bayes factor) (relative likelihood)
        self.results['log10_BF_BIC'] = - delta_BIC/2 / np.log(10) # Calculate log10(Bayes factor) (relative likelihood)
        
        self.results['best_model_AIC'] = (self.results.AIC == np.min(self.results.AIC)).astype(int)
        self.results['best_model_BIC'] = (self.results.BIC == np.min(self.results.BIC)).astype(int)
        
        self.results_sort = self.results.sort_values(by='AIC')
        
        self.trial_numbers = result_this.trial_numbers 
        
        # == Plotting == 
        if plot_predictive is not None: # Plot the predictive choice trace of the best fitting of the best model (Using AIC)
            self.plot_predictive = plot_predictive
            self.plot_predictive_choice()
        return
    
    def cross_validate(self, k_fold = 2, fit_method = 'DE', fit_settings = {'DE_pop_size': 16}, pool = '', if_verbose = True):
        
        self.prediction_accuracy_CV = pd.DataFrame()
        
        if if_verbose: print('=== Cross validation ===\nMethods = %s, %s, pool = %s' % (fit_method, fit_settings, pool!=''))
        
        for mm, model in enumerate(self.models):
            # == Get settings for this model ==
            forager, fit_names, fit_lb, fit_ub = model
            fit_bounds = [fit_lb, fit_ub]
            
            para_notation = ''
            Km = 0
            
            for name, lb, ub in zip(fit_names, fit_lb, fit_ub):
                # == Generate notation ==
                if lb < ub:
                    para_notation += PARA_NOTATIONS[name] + ', '
                    Km += 1
            
            para_notation = para_notation[:-2]
            
            # == Do fitting here ==
            #  Km = np.sum(np.diff(np.array(fit_bounds),axis=0)>0)
            
            if if_verbose: print('Model %g/%g: %15s, Km = %g ...'%(mm+1, len(self.models), forager, Km), end = '')
            start = time.time()
                
            prediction_accuracy_test, prediction_accuracy_fit, prediction_accuracy_test_bias_only= cross_validate_bandit(forager, fit_names, fit_bounds, 
                                                                                      self.fit_choice_history, self.fit_reward_history, self.session_num, 
                                                                                      k_fold = k_fold, **fit_settings, pool = pool, if_verbose = if_verbose) #plot_predictive is not None)
            
            if if_verbose: print('  \n%g-fold CV: Test acc.= %s, Fit acc. = %s (done in %.3g secs)' % (k_fold, prediction_accuracy_test, prediction_accuracy_fit, time.time()-start) )
            
            self.prediction_accuracy_CV = pd.concat([self.prediction_accuracy_CV, 
                                                     pd.DataFrame({'model#': mm,
                                                                   'forager': forager,
                                                                   'Km': Km,
                                                                   'para_notation': para_notation,
                                                                   'prediction_accuracy_test': prediction_accuracy_test, 
                                                                   'prediction_accuracy_fit': prediction_accuracy_fit,
                                                                   'prediction_accuracy_test_bias_only': prediction_accuracy_test_bias_only})])
            
        return

    def plot_predictive_choice(self):
        plot_model_comparison_predictive_choice_prob(self)

    def show(self):
        pd.options.display.max_colwidth = 100
        display(self.results_sort[['model','Km', 'AIC','log10_BF_AIC', 'model_weight_AIC', 'BIC','log10_BF_BIC', 'model_weight_BIC', 'para_notation','para_fitted']].round(2))
        
    def plot(self):
        plot_model_comparison_result(self)


import numpy as np
import seaborn as sns
import matplotlib
from scipy.optimize import curve_fit


def softmax(x, softmax_temperature, bias = 0):
    
    # Put the bias outside /sigma to make it comparable across different softmax_temperatures.
    if len(x.shape) == 1:
        X = x/softmax_temperature + bias   # Backward compatibility
    else:
        X = np.sum(x/softmax_temperature, axis=0) + bias  # Allow more than one kernels (e.g., choice kernel)
    
    max_temp = np.max(X)
    
    if max_temp > 700: # To prevent explosion of EXP
        greedy = np.zeros(len(x))
        greedy[np.random.choice(np.where(X == np.max(X))[0])] = 1
        return greedy
    else:   # Normal softmax
        return np.exp(X)/np.sum(np.exp(X))  # Accept np.
    
# def choose_ps(ps):
#     '''
#     "Poisson"-choice process
#     '''
#     ps = ps/np.sum(ps)
#     return np.max(np.argwhere(np.hstack([-1e-16, np.cumsum(ps)]) < np.random.rand()))

def seaborn_style():
    """
    Set seaborn style for plotting figures
    """
    sns.set(style="ticks", context="paper", font_scale=1.4)
    # sns.set(style="ticks", context="talk", font_scale=2)
    sns.despine(trim=True)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
def moving_average(a, n=3) :
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def fit_sigmoid_p_choice(p_reward, choice, win=10, stepsize=None):
    if stepsize is None: stepsize = win
    start_trial = 0
    mean_p_diff = []
    mean_choice_R_frac = []

    while start_trial + win <= len(choice):
        end_trial = start_trial + win
        
        mean_p_diff.append(np.mean(np.diff(p_reward[:, start_trial:end_trial], axis=0)))
        mean_choice_R_frac.append(np.sum(choice[start_trial:end_trial] == 1) / win)
        
        start_trial += stepsize
        
    mean_p_diff = np.array(mean_p_diff)
    mean_choice_R_frac = np.array(mean_choice_R_frac)

    p0 = [0, 1, 1, 0]

    popt, pcov = curve_fit(lambda x, x0, k: sigmoid(x, x0, k, a=1, b=0), 
                        mean_p_diff, 
                        mean_choice_R_frac, 
                        p0[:2], 
                        method='lm',
                        maxfev=10000)
    
    return popt, pcov, mean_p_diff, mean_choice_R_frac

def sigmoid(x, x0, k, a, b):
    y = a / (1 + np.exp(-k * (x - x0))) + b
    return y





# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:15:59 2020

@author: Han
"""

import multiprocessing as mp
# import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys

#from models.bandit_model_comparison import BanditModelComparison, MODELS
#from models.fitting_functions import fit_bandit, negLL_func
   
def fit_para_recovery(forager, para_names, para_bounds, true_paras = None, n_models = 10, n_trials = 1000, 
                      para_scales = None, para_color_code = None, para_2ds = [[0,1]], fit_method = 'DE', DE_pop_size = 16, n_x0s = 1, pool = '', **kwargs):
    
    n_paras = len(para_names)
    
    if true_paras is None:
        if_no_true_paras = True
        true_paras = np.zeros([n_paras, n_models])
    else:
        if_no_true_paras = False
        n_models = np.shape(true_paras)[1]
        
    fitted_paras = np.zeros([n_paras, n_models])

    # === Do para recovery ===        
    for n in tqdm(range(n_models), desc='Parameter Recovery, %s'%forager):
        # Generate simulated para using uniform distribution in para_bounds if not specified
        if if_no_true_paras: 
            true_paras_this = []
            for pp in range(n_paras):
                true_paras_this.append(np.random.uniform(para_bounds[0][pp], para_bounds[1][pp]))
            true_paras[:,n] = true_paras_this

        choice_history, reward_history, _ = generate_fake_data(forager, para_names, true_paras[:,n], **{'n_trials': n_trials,**kwargs})
        fitting_result = fit_bandit(
            forager, para_names, para_bounds, choice_history, reward_history,
            fit_method=fit_method, DE_pop_size=DE_pop_size, n_x0s=n_x0s, pool=pool)
        fitted_paras[:,n] = fitting_result.x
    
            # print(true_paras_this, fitting_result.x)
        
    # === Plot results ===
    if fit_method == 'DE':
        fit_method = 'DE ' + '(pop_size = %g)' % DE_pop_size
    else:
        fit_method = fit_method + ' (n_x0s = %g)' % n_x0s
    
    plot_para_recovery(forager, true_paras, fitted_paras, para_names, para_bounds, para_scales, para_color_code, para_2ds, n_trials, fit_method)
    
    return true_paras, fitted_paras

    
def generate_fake_data(forager, para_names, true_para, n_trials = 1000, **kwargs):
    # Generate fake data
    n_paras = len(para_names)
    kwarg_this = {}
    for pp in range(n_paras):
        kwarg_this[para_names[pp]] = true_para[pp]
    
    bandit = BanditModel(forager, n_trials = n_trials, **kwarg_this, **kwargs)
    bandit.simulate()
    
    choice_history = bandit.choice_history
    reward_history = bandit.reward_history
    schedule = bandit.p_reward 
    
    return choice_history, reward_history, schedule


def compute_LL_surface(forager, para_names, para_bounds, true_para, 
                       para_2ds = [[0,1]], n_grids = None, para_scales = None, 
                       fit_method = 'DE', DE_pop_size = 16, n_x0s = 1, pool = '', n_trials = 1000, **kwargs):
    '''
    Log-likelihood landscape (Fig.3a, Wilson 2019)

    '''

    # Backward compatibility
    if para_scales is None: 
        para_scales = ['linear'] * len(para_names)
    if n_grids is None:
        n_grids = [[20,20]] * len(para_names)
    
    n_worker = int(mp.cpu_count())
    pool_surface = mp.Pool(processes = n_worker)
    para_grids = []
    
    # === 1. Generate fake data; make sure the true_paras are exactly on the grid ===
    for para_2d, n_g in zip(para_2ds, n_grids):
        
        if para_scales[para_2d[0]] == 'linear':
            p1 = np.linspace(para_bounds[0][para_2d[0]], para_bounds[1][para_2d[0]], n_g[0])
        else:
            p1 = np.logspace(np.log10(para_bounds[0][para_2d[0]]), np.log10(para_bounds[1][para_2d[0]]), n_g[0])
            
        if para_scales[para_2d[1]] == 'linear':
            p2 = np.linspace(para_bounds[0][para_2d[1]], para_bounds[1][para_2d[1]], n_g[1])
        else:
            p2 = np.logspace(np.log10(para_bounds[0][para_2d[1]]), np.log10(para_bounds[1][para_2d[1]]), n_g[1])
            
        # -- Don't do this --
        # Make sure the true_paras are exactly on the grid
        # true_para[para_2d[0]] = p1[np.argmin(np.abs(true_para[para_2d[0]] - p1))]
        # true_para[para_2d[1]] = p2[np.argmin(np.abs(true_para[para_2d[1]] - p2))]
        
        # Save para_grids
        para_grids.append([p1, p2])
        
    # === 3. Generate fake data using the adjusted true value ===
    # print('Adjusted true para on grid: %s' % np.round(true_para,3))
    choice_history, reward_history, p_reward = generate_fake_data(forager, para_names, true_para, n_trials, **kwargs)
    session_num = np.zeros_like(choice_history)[0]  # Regard as one session

    # === 4. Do fitting only once ===
    if fit_method == 'DE':
        print('Fitting using %s (pop_size = %g), pool = %s...'%(fit_method, DE_pop_size, pool!=''))
    else:
        print('Fitting using %s (n_x0s = %g), pool = %s...'%(fit_method, n_x0s, pool!=''))
    
    fitting_result = fit_bandit(forager, para_names, para_bounds, choice_history, reward_history, 
                                             fit_method = fit_method, DE_pop_size = DE_pop_size, n_x0s = n_x0s, pool = pool,
                                             if_history = True, if_predictive = True)
    
    print('  True para: %s' % np.round(true_para,3))
    print('Fitted para: %s' % np.round(fitting_result.x,3))
    print('km = %g, AIC = %g, BIC = %g\n      LPT_AIC = %g, LPT_BIC = %g' % (fitting_result.k_model, np.round(fitting_result.AIC, 3), np.round(fitting_result.BIC, 3),
                                                                             np.round(fitting_result.LPT_AIC, 3), np.round(fitting_result.LPT_BIC, 3)))
    sys.stdout.flush()
       
    # === 5. Plot fitted curve ===
    plot_session_lightweight([choice_history, reward_history, p_reward], fitting_result.predictive_choice_prob)

    # === 6. Compute LL surfaces for all pairs ===
    LLsurfaces = []
    CI_cutoff_LPTs = []
    
    for ppp,((p1, p2), n_g, para_2d) in enumerate(zip(para_grids, n_grids, para_2ds)):
           
        pp2, pp1 = np.meshgrid(p2,p1)  # Note the order

        n_scan_paras = np.prod(n_g)
        LLs = np.zeros(n_scan_paras)
        
        # Make other parameters fixed at the fitted value
        para_fixed = {}
        for p_ind, (para_fixed_name, para_fixed_value) in enumerate(zip(para_names, fitting_result.x)):
            if p_ind not in para_2d: 
                para_fixed[para_fixed_name] = para_fixed_value
        
        # -- In parallel --
        pool_results = []
        for x,y in zip(np.nditer(pp1),np.nditer(pp2)):
            pool_results.append(pool_surface.apply_async(negLL_func, args = ([x, y], forager, [para_names[para_2d[0]], para_names[para_2d[1]]], choice_history, reward_history, session_num, para_fixed, [])))
            
        # Must use two separate for loops, one for assigning and one for harvesting!   
        for nn,rr in tqdm(enumerate(pool_results), total = n_scan_paras, desc='LL_surface pair #%g' % ppp):
            LLs[nn] = - rr.get()
            
        # -- Serial for debugging --
        # for nn,(x,y) in tqdm(enumerate(zip(np.nditer(pp1),np.nditer(pp2))), total = n_scan_paras, desc='LL_surface pair #%g (serial)' % ppp):
        #     LLs[nn] = negLL_func([x, y], forager, [para_names[para_2d[0]], para_names[para_2d[1]]], choice_history, reward_history, session_num, para_fixed, [])
        
        # -- Confidence interval --
        # https://www.umass.edu/landeco/teaching/ecodata/schedule/likelihood.pdf
        CI_cutoff_LL = np.max(LLs) - 3
        CI_cutoff_LPT = np.exp(CI_cutoff_LL / n_trials)
        CI_cutoff_LPTs.append(CI_cutoff_LPT)
        
        LLs = np.exp(LLs/n_trials)  # Use likelihood-per-trial = (likehood)^(1/T)
        LLs = LLs.reshape(n_g).T
        LLsurfaces.append(LLs)
 
    
    pool_surface.close()
    pool_surface.join()

    
    # Plot LL surface and fitting history
    if fit_method == 'DE':
        fit_method = 'DE ' + '(pop_size = %g)' % DE_pop_size
    else:
        fit_method = fit_method + ' (n_x0s = %g)'%n_x0s

    plot_LL_surface(forager, LLsurfaces, CI_cutoff_LPTs, para_names, para_2ds, para_grids, para_scales, true_para, fitting_result.x, fitting_result.fit_histories, fit_method, n_trials)
    
    return

def compute_confusion_matrix(models = [1,2,3,4,5,6,7,8], n_runs = 2, n_trials = 1000, pool = '', save_file = '', save_folder = '..\\results\\'):
    if models is None:  
        models = MODELS
    elif type(models[0]) is int:
        models = [MODELS[i-1] for i in models]

    n_models = len(models)
    confusion_idx = ['AIC', 'BIC', 'log10_BF_AIC', 'log10_BF_BIC', 'best_model_AIC', 'best_model_BIC']
    confusion_results = {}
    
    confusion_results['models'] = models
    confusion_results['n_runs'] = n_runs
    confusion_results['n_trials'] = n_trials
    
    for idx in confusion_idx:
        confusion_results['raw_' + idx] = np.zeros([n_models, n_models, n_runs])
        confusion_results['raw_' + idx][:] = np.nan
    
    # == Simulation ==
    for rr in tqdm(range(n_runs), total = n_runs, desc = 'Runs'):
        for mm, this_model in enumerate(models):
            this_forager, this_para_names = this_model[0], this_model[1]
            
            # Generate para
            this_true_para = []
            for pp in this_para_names:
                this_true_para.append(generate_random_para(this_forager, pp))
            
            # Generate fake data
            choice_history, reward_history, p_reward = generate_fake_data(this_forager, this_para_names, this_true_para, n_trials = n_trials)
            
            # Do model comparison
            model_comparison = BanditModelComparison(choice_history, reward_history, p_reward , models = models)
            model_comparison.fit(pool = pool, if_verbose = False)
            
            # Save data
            for idx in confusion_idx:
                confusion_results['raw_' + idx][mm, :, rr] = model_comparison.results[idx]
    
        # == Average across runs till now ==
        for idx in confusion_idx:
            confusion_results['confusion_' + idx] = np.nanmean(confusion_results['raw_' + idx], axis = 2)
        
        # == Compute inversion matrix ==
        confusion_results['inversion_best_model_AIC'] = confusion_results['confusion_best_model_AIC'] / (1e-10 + np.sum(confusion_results['confusion_best_model_AIC'], axis = 0)) 
        confusion_results['inversion_best_model_BIC'] = confusion_results['confusion_best_model_BIC'] / (1e-10 + np.sum(confusion_results['confusion_best_model_BIC'], axis = 0))
        
        # == Save data (after each run) ==
        confusion_results['models_notations'] = model_comparison.results.para_notation
        if save_file == '':
            save_file = "confusion_results_%s_%s.p" % (n_runs, n_trials)
        
        pickle.dump(confusion_results, open(save_folder+save_file, "wb"))
        
    return
        
def generate_random_para(forager, para_name):
    # With slightly narrower range than fitting bounds in BanditModelComparison
    if para_name in 'loss_count_threshold_mean':
        return np.random.uniform(0, 30)
    if para_name in 'loss_count_threshold_std':
        return np.random.uniform(0, 5)
    if para_name in ['tau1', 'tau2']:
        return 10**np.random.uniform(0, np.log10(30)) 
    if para_name in ['w_tau1', 'learn_rate', 'learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'epsilon']:
        return np.random.uniform(0, 1)
    if para_name in 'softmax_temperature':
        return 1/np.random.exponential(10)
    if para_name in ['biasL']:
        if forager in ['Random', 'pMatching', 'RW1972_epsi']:
            return np.random.uniform(-0.45, 0.45)
        elif forager in ['RW1972_softmax', 'LNP_softmax', 'Bari2019', 'Hattori2019']:
            return np.random.uniform(-5, 5)
    return np.nan    
    

# %%

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class BaseHandler(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        pass


class TrajAnalysisPrintDebug(BaseHandler):
    def compute_states_outputs_derivs(self, system):
        traj_analysis = self.model_instance.implementation
        self.traj_analysis = traj_analysis

        res = system.result

        # system.result doesn't save off dynamic_ouptuts by default,
        # so generate them here
        output_hist = np.empty(
            (np.array(res.t).size, traj_analysis.model.dynamic_output._count)
        )
        # create history of state derivatives
        dot_hist = np.empty(np.array(res.x).shape)
        if traj_analysis.dynamic_output_func:
            for idx, (t, x) in enumerate(zip(res.t, res.x)):
                output_hist[idx, None] = traj_analysis.dynamic_output_func(
                    res.p, t, x
                ).T
                dot_hist[idx, None] = system.dots(res.t[idx], np.array(res.x)[idx, :])

        res.y = output_hist

        # convert to np arrays for bind_result function
        res.x = np.array(res.x)
        res.y = np.array(res.y)

        self.traj_analysis.bind_result(traj_analysis.model_instance, res)
        self.dot_hist = self.traj_analysis.model.state.wrap(dot_hist.T)

    def __call__(self, **kwargs):
        system = kwargs.pop("system", None)

        self.compute_states_outputs_derivs(system)
        traj_analysis = self.traj_analysis

        print("\n\nSimulation Time Before Failure: ")
        print(system.result.t[-1])

        print("\n\nValues of States Before Failure: ")
        for name in traj_analysis.model_instance.state.asdict():
            data = traj_analysis.model_instance.state.asdict()[name]
            print(f"\nValue of {name} before integration failure:")
            print(data[..., -1])

        print("\n\nValues of State Derivatives Before Failure: ")
        for name in traj_analysis.model_instance.state.asdict():
            data = self.dot_hist.asdict()[name]
            print(f"\nValue of {name} derivative before integration failure:")
            print(data[..., -1])

        print("\n\nValues of Outputs Before Failure: ")
        for name in traj_analysis.model_instance.dynamic_output.asdict():
            data = traj_analysis.model_instance.dynamic_output.asdict()[name]
            print(f"\nValue of {name} before integration failure:")
            print(data[..., -1])


class TrajAnalysisSaveNpzHandler(BaseHandler):
    def __init__(self, path="trajectory_analysis_result_log"):
        self.debug_file_name = path

    def __call__(self, **kwargs):
        print(f"Saving results to {self.debug_file_name}")
        system = kwargs.pop("system", None)
        system.result.save(self.debug_file_name)


class BreakpointHandler(BaseHandler):
    def __call__(self, **kwargs):
        breakpoint()


class ExceptionHandler(BaseHandler):
    def __call__(self, **kwargs):
        error_msg = "Integration unsuccessful"
        raise RuntimeError(error_msg)


class SolverHandlerEnum(Enum):
    BASE = BaseHandler
    TRAJANALYSISPRINTDEBUG = TrajAnalysisPrintDebug
    TRAJANALYSISSAVENPZ = TrajAnalysisSaveNpzHandler
    BREAKPOINT = BreakpointHandler
    EXCEPTION = ExceptionHandler


def create_solver_failure_handler(model_instance, failure_handler_choices):
    if failure_handler_choices is not None:
        stacked_handlers = []

        if not isinstance(failure_handler_choices, list):
            failure_handler_choices = [failure_handler_choices]

        for handle in failure_handler_choices:
            if isinstance(handle, str):
                # This handle string has to be member of SolverHandlerEnum
                handler_class = SolverHandlerEnum[handle.upper()].value
                handler = handler_class()
                handler.model_instance = model_instance
                stacked_handlers.append(handler)

            elif callable(handle):
                handle.model_instance = model_instance
                stacked_handlers.append(handle)

        def stacked_callable(**kwargs):
            for func in stacked_handlers:
                func(**kwargs)

        handler = stacked_callable
    else:
        handler = None

    return handler

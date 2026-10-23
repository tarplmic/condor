from enum import StrEnum

import numpy as np


class SolverHandlerEnum(StrEnum):
    BASE = "basehandler"
    PRINTDEBUG = "printdebughandler"
    SAVENPZ = "savenpzhandler"
    BREAK = "breakpointhandler"
    EXCEPT = "exceptionhandler"

    @property
    def handler_class(self):
        mapping = {
            SolverHandlerEnum.BASE: BaseHandler,
            SolverHandlerEnum.PRINTDEBUG: PrintDebugHandler,
            SolverHandlerEnum.SAVENPZ: SaveNpzHandler,
            SolverHandlerEnum.BREAK: BreakpointHandler,
            SolverHandlerEnum.EXCEPT: ExceptionHandler,
        }
        return mapping[self]


class BaseHandler:
    def __init__(self):
        pass

    def set_model(self, model):
        self.traj_analysis = model

    def __call__(self, system):
        pass


class PrintDebugHandler(BaseHandler):
    def __init__(self):
        pass

    def compute_states_outputs_derivs(self, system):
        traj_analysis = self.traj_analysis

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

    def __call__(self, system):
        self.compute_states_outputs_derivs(system)
        traj_analysis = self.traj_analysis

        print("\n\nValues of States Before Failure: ")
        for name in traj_analysis.model_instance.state.asdict():
            data = traj_analysis.model_instance.state.asdict()[name]
            print(f"\nValue of {name} before integration failure:")
            print(data[:, :, -1])

        print("\n\nValues of Outputs Before Failure: ")
        for name in traj_analysis.model_instance.dynamic_output.asdict():
            data = traj_analysis.model_instance.dynamic_output.asdict()[name]
            print(f"\nValue of {name} before integration failure:")
            print(data[:, :, -1])


class SaveNpzHandler(BaseHandler):
    def __init__(self, path="trajectory_analysis_result_log"):
        self.debug_file_name = path

    def __call__(self, system):
        print(f"Saving results to {self.debug_file_name}")
        system.result.save(self.debug_file_name)


class BreakpointHandler(BaseHandler):
    def __init__(self):
        pass

    def __call__(self, system):
        breakpoint()


class ExceptionHandler(BaseHandler):
    def __init__(self):
        pass

    def __call__(self, system):
        error_msg = "Integration unsuccessful."
        raise RuntimeError(error_msg)

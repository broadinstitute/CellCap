"""Training plan with arbitrary logging"""

from scvi.train._trainingplans import TrainingPlan
from .utils import _METRICS_TO_LOG


# TODO: maybe not worth it to figure out this commented-out approach
# def training_epoch_end(cls, outputs):
#     cls.training_epoch_end(outputs=outputs)
#     for name, val in _METRICS_TO_LOG:
#         cls.log(name, val)
#
#
# def logging_wrapper(superclass):
#     """
#     Programmatically create subclasses which overwrite the
#     training_epoch_end() method.
#
#     Examples
#     --------
#
#     >>> training_plan = LoggedTrainingPlan(self.module, **plan_kwargs)
#     would be the same as
#     >>> training_plan = logging_wrapper(TrainingPlan)(self.module, **plan_kwargs)
#     except now you can also do
#     >>> training_plan = logging_wrapper(MyTrainingPlan)(self.module, **plan_kwargs)
#     """
#
#     _LoggedClass = type(f'Logged{superclass}', superclass, {})
#     _LoggedClass.training_epoch_end = classmethod(training_epoch_end)
#
#     return _LoggedClass


class LoggedTrainingPlan(TrainingPlan):

    def training_epoch_end(self, outputs):
        super().training_epoch_end(outputs=outputs)
        for name, val in _METRICS_TO_LOG:
            self.log(name, val)

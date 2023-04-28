from abc import abstractmethod
from typing import Any, Dict, Optional

from mesonic.synth import Synth
from pandas import DataFrame

from sonecules.base import Sonecule

# TODO rename modules to scorebased son, ... depending on thier usage


class StandardPMSon(Sonecule):
    def __init__(
        self,
        synth: str = "s2",
        parameter_specs: Optional[Dict[str, Dict[str, Any]]] = None,
        context=None,
    ):
        super().__init__(context)
        self._pepare_synth(synth)

        if parameter_specs is None:
            parameter_specs = {}

        # treat the rest f.e. bounds as synth attributes
        for param, param_spec in parameter_specs.items():
            if param not in self._synth.params:
                raise ValueError(f"{param} is not a Parameter of {self._synth}")
            if "bounds" in param_spec:
                getattr(self._synth, param).bounds = param_spec["bounds"]
            if "default" in param_spec:
                getattr(self._synth, param)._default = param_spec["default"]

        # TODO Synth creation vs passing
        # creation here should add metadata which then is added to each produced event
        self._synth.metadata = self.sonecule_id

    @abstractmethod
    def _pepare_synth(self, synth):
        ...

    def synth(self):
        self._synth

    @abstractmethod
    def schedule(
        self,
        df: DataFrame,
        mapping,
        at=0,
        stop_after=0.1,
        sort_by_onset=True,
        **odfkwargs,
    ):  # odfkwargs is a bad name
        # stop after =~~~ release time
        # clear the current events from the timeline
        ...


class StandardContinuousPMSon(StandardPMSon):
    def _pepare_synth(self, synth):
        if isinstance(synth, Synth):
            assert (
                synth.mutable
            ), "Synth needs to be mutable for continuous Parameter Mapping Sonification"
            self._synth = synth
        else:
            self._synth = self.context.synths.create(synth, track=1)

    def schedule(
        self,
        df: DataFrame,
        mapping,
        at=0,
        stop_after=0.1,
        sort_by_onset=True,
        **odfkwargs,
    ):
        self.reset()

        if sort_by_onset:
            col, _, _ = mapping["onset"]
            df.sort_values(by=col, ascending=True, inplace=True)

        dfkwargs = {"dmin": df.min(), "dmax": df.max()}  # TODO names for dmin, dmax?
        dfkwargs.update(odfkwargs)  # allow overwriting of df

        with self._context.at(at, info={"sonecule_id": self.sonecule_id}):
            self._synth.start()

        max_onset = 0
        for idx in df.index:
            col, fun, mkwargs = mapping["onset"]
            value = getattr(df, col)[idx]

            # get column wise df kwargs
            data_min, data_max = dfkwargs["dmin"][col], dfkwargs["dmax"][col]

            onset = fun(value, **mkwargs, dmin=data_min, dmax=data_max)
            max_onset = max(onset, max_onset)

            params = {}
            for param in [param for param in mapping.keys() if param != "onset"]:
                col, fun, mkwargs = mapping[param]
                value = getattr(df, col)[idx]

                dmin, dmax = dfkwargs["dmin"][col], dfkwargs["dmax"][col]

                params[param] = fun(value, **mkwargs, dmin=dmin, dmax=dmax)

            with self._context.at(at + onset, info={"sonecule_id": self.sonecule_id}):
                self._synth.set(params=params)

        with self._context.at(
            at + max_onset + stop_after, info={"sonecule_id": self.sonecule_id}
        ):
            self._synth.stop()


class StandardDiscretePMSon(StandardPMSon):
    def _pepare_synth(self, synth):
        if isinstance(synth, Synth):
            assert (
                not synth.mutable
            ), "Synth needs to be mutable for continuous Parameter Mapping Sonification"
            self._synth = synth
        else:
            self._synth = self.context.synths.create(synth, track=1, mutable=False)

    def schedule(
        self,
        df: DataFrame,
        mapping,
        at=0,
        stop_after=0.1,
        sort_by_onset=True,
        **odfkwargs,
    ):
        self.reset()

        if sort_by_onset:
            col, _, _ = mapping["onset"]
            df.sort_values(by=col, ascending=True, inplace=True)

        dfkwargs = {"dmin": df.min(), "dmax": df.max()}  # TODO names for dmin, dmax?
        dfkwargs.update(odfkwargs)  # allow overwriting of df

        for idx in df.index:
            col, fun, mkwargs = mapping["onset"]
            value = getattr(df, col)[idx]

            # get column wise df kwargs
            data_min, data_max = dfkwargs["dmin"][col], dfkwargs["dmax"][col]

            onset = fun(value, **mkwargs, dmin=data_min, dmax=data_max)

            params = {}
            for param in [param for param in mapping.keys() if param != "onset"]:
                col, fun, mkwargs = mapping[param]
                value = getattr(df, col)[idx]

                dmin, dmax = dfkwargs["dmin"][col], dfkwargs["dmax"][col]

                params[param] = fun(value, **mkwargs, dmin=dmin, dmax=dmax)

            with self._context.at(at + onset, info={"sonecule_id": self.sonecule_id}):
                self._synth.start(params=params)

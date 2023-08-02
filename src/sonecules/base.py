import uuid
from abc import ABC, abstractmethod
from typing import Optional

from mesonic.context import Context
from mesonic.events import Event

from sonecules import Sonecules


class Sonecule(ABC):
    def __init__(self, context: Optional[Context] = None, sonecule_id=None):
        if context is None:
            context = Sonecules.default_context
        self._context: Context = context
        self._sonecule_id = uuid.uuid4() if sonecule_id is None else sonecule_id
        self._prepare_synth_defs()
        # TODO require or better ensure each synth created in a sonecule gets the
        # metadata set to include the sonecule_id so it can be filtere

        # TODO ensure self._context.processor.event_filters has a sonecule filter
        # needed for active

    @property
    def context(self):
        return self._context

    @property
    def sonecule_id(self):
        return self._sonecule_id

    @property
    def active(self):
        return self.sonecule_id not in Sonecules.event_filter.deactivated_sonecules

    @active.setter
    def active(self, active):
        assert isinstance(active, bool)
        if active:  # this sonecule should not be part of the timeline
            Sonecules.event_filter.deactivated_sonecules.discard(self.sonecule_id)
        else:
            Sonecules.event_filter.deactivated_sonecules.add(self.sonecule_id)

    def remove(self):
        """Remove events belonging to this Sonecule from the mesonic Timeline"""

        def sonecule_id_filter(event: Event) -> bool:
            return event.info.get("sonecule_id", None) == self._sonecule_id

        self._context.timeline.filter(sonecule_id_filter)

    def start(self, **kwargs):
        """start sonification rendering by starting the playback
        kwargs are passed on to start(), so use rate to control speedup, etc.
        """
        self.context.playback.start(**kwargs)
        return self

    @abstractmethod
    def _prepare_synth_defs(self):
        ...


class SchedulableSonecule(Sonecule):
    @abstractmethod
    def schedule(self, at: float = 0.0, **kwargs):
        """Schedule the Sonification.

        Parameters
        ----------
        at : float, optional
            the start time of the Sonification, by default 0.0

        Returns
        -------
        self
        """
        ...

    def reschedule(self, at: float = 0.0, **kwargs):
        """Reschedule the Sonification.

        reschedule = remove + schedule

        Parameters
        ----------
        at : float, optional
            the start time of the Sonification, by default 0.0

        Returns
        -------
        self
        """
        self.remove()
        self.schedule(at=at, **kwargs)
        return self

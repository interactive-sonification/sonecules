import uuid
from typing import Optional

from mesonic.context import Context

from sonecules import Sonecules


class Sonecule:
    def __init__(self, context: Optional[Context] = None):
        if context is None:
            context = Sonecules.default_context
        self._context: Context = context
        self._sonecule_id: uuid.UUID = uuid.uuid4()
        # TODO require or better ensure each synth created in a sonecule gets the
        # metadata set to include the sonecule_id so it can be filtere

        # TODO ensure self._context.processor.event_filter has a sonecule filter
        # needed for active

    @property
    def context(self):
        return self._context

    @property
    def sonecule_id(self):
        return self._sonecule_id

    @property
    def active(self):
        return (
            self.sonecule_id
            in self._context.processor.event_filter.deactivated_sonecules
        )

    @active.setter
    def active(self, value):
        assert isinstance(value, bool)
        if value:  # this sonecule should not be part of the timeline
            self._context.processor.event_filter.deactivated_sonecules.add(
                self.sonecule_id
            )
        else:
            self._context.processor.event_filter.deactivated_sonecules.discard(
                self.sonecule_id
            )

    def reset(self):
        """Remove events belonging to this Sonecule from the mesonic Timeline"""
        # TODO this should be done in mesonic
        # self._context.timeline.
        pass  # raise NotImplementedError

"""Entity module for luckylab.

Provides Entity, EntityData, and Scene classes that match mjlab's API
but work with LuckyEngine observations.
"""

from luckylab.entity.data import EntityData as EntityData
from luckylab.entity.data import ObservationSchema as ObservationSchema
from luckylab.entity.entity import Entity as Entity
from luckylab.entity.entity import EntityCfg as EntityCfg
from luckylab.entity.scene import Scene as Scene

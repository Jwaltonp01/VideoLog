from enum import Enum


class Rank(Enum):
    One = "Rank_1"
    Two = "Rank_2"
    Three = "Rank_3"
    Four = "Rank_4"
    Five = "Rank_5"
    NG = "Rank_NG"


class ModelType(Enum):
    Rank = "rank"
    Stat = "stat"
    VidLog = "vidlog"


class Response(Enum):
    SUCCESS = 201
    UNKNOWN_ERROR = 402
    FILE_NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500

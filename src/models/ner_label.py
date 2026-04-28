from enum import StrEnum


class NerLabel(StrEnum):
    ART    = "ART"
    DATE   = "DATE"
    DOC    = "DOC"
    JOB    = "JOB"
    LOC    = "LOC"
    MON    = "MON"
    ORG    = "ORG"
    PCT    = "PCT"
    PERIOD = "PERIOD"
    PERS   = "PERS"
    QUANT  = "QUANT"
    TIME   = "TIME"
    MISC   = "MISC"

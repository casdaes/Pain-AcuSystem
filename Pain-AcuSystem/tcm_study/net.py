
from __future__ import annotations

import os
import socket

_PATCHED = False


def force_ipv4() -> None:
    global _PATCHED
    if _PATCHED or os.environ.get("TCM_FORCE_IPV4", "1") == "0":
        return
    orig = socket.getaddrinfo

    def getaddrinfo_v4(host, port, family=0, type=0, proto=0, flags=0):
        if family in (0, socket.AF_UNSPEC):
            try:
                res = orig(host, port, socket.AF_INET, type, proto, flags)
                if res:
                    return res
            except socket.gaierror:
                pass                                  # 只有 IPv6 地址的主机：退回系统默认
        return orig(host, port, family, type, proto, flags)

    socket.getaddrinfo = getaddrinfo_v4
    _PATCHED = True

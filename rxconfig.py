from typing import Any, cast

import reflex as rx

rx = cast(Any, rx)  # pyre-ignore[31]

config = rx.Config(
    app_name="harness",
    app_module_import="harness",
    disable_plugins=["reflex.plugins.sitemap.SitemapPlugin"],
    state_auto_setters=True,
    loglevel=rx.constants.LogLevel.CRITICAL,
)

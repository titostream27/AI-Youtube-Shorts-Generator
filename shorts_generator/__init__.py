"""Shorts generator package (media execution worker).

Phase 2: the standalone/API-mode highlight pipeline was removed. The
production path is the render service (render_service.py), which uses
shorts_generator.local.* for cutting + reframing. No virality ranking or
highlight discovery lives in this repository.
"""

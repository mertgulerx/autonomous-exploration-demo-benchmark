# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes as ct
import json
import sys


def probe() -> dict[str, str]:
    egl = ct.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [ct.c_char_p]
    egl.eglGetProcAddress.restype = ct.c_void_p
    query_address = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
    display_address = egl.eglGetProcAddress(b"eglGetPlatformDisplayEXT")
    if not query_address or not display_address:
        raise RuntimeError("EGL device enumeration is unavailable")
    query = ct.CFUNCTYPE(ct.c_uint, ct.c_int, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_int))(
        query_address
    )
    get_display = ct.CFUNCTYPE(ct.c_void_p, ct.c_uint, ct.c_void_p, ct.POINTER(ct.c_int))(
        display_address
    )
    egl.eglInitialize.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
    egl.eglInitialize.restype = ct.c_uint
    egl.eglChooseConfig.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_void_p),
        ct.c_int,
        ct.POINTER(ct.c_int),
    ]
    egl.eglChooseConfig.restype = ct.c_uint
    egl.eglBindAPI.argtypes = [ct.c_uint]
    egl.eglBindAPI.restype = ct.c_uint
    egl.eglCreatePbufferSurface.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_int)]
    egl.eglCreatePbufferSurface.restype = ct.c_void_p
    egl.eglCreateContext.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_int)]
    egl.eglCreateContext.restype = ct.c_void_p
    egl.eglMakeCurrent.argtypes = [ct.c_void_p] * 4
    egl.eglMakeCurrent.restype = ct.c_uint
    egl.eglDestroyContext.argtypes = [ct.c_void_p, ct.c_void_p]
    egl.eglDestroySurface.argtypes = [ct.c_void_p, ct.c_void_p]
    egl.eglTerminate.argtypes = [ct.c_void_p]
    devices = (ct.c_void_p * 32)()
    count = ct.c_int()
    if not query(len(devices), devices, ct.byref(count)) or count.value == 0:
        raise RuntimeError("No EGL devices are accessible inside the container")
    errors = []
    for device in devices[: count.value]:
        display = get_display(0x313F, device, None)
        major, minor = ct.c_int(), ct.c_int()
        if not display or not egl.eglInitialize(display, ct.byref(major), ct.byref(minor)):
            errors.append("EGL display initialization failed")
            continue
        context = surface = None
        try:
            attributes = (ct.c_int * 5)(0x3033, 1, 0x3040, 8, 0x3038)
            config, matches = ct.c_void_p(), ct.c_int()
            if (
                not egl.eglChooseConfig(display, attributes, ct.byref(config), 1, ct.byref(matches))
                or not matches.value
            ):
                raise RuntimeError("No EGL OpenGL pbuffer configuration")
            if not egl.eglBindAPI(0x30A2):
                raise RuntimeError("EGL cannot bind the OpenGL API")
            size = (ct.c_int * 5)(0x3057, 1, 0x3056, 1, 0x3038)
            surface = egl.eglCreatePbufferSurface(display, config, size)
            version = (ct.c_int * 5)(0x3098, 3, 0x30FB, 3, 0x3038)
            context = egl.eglCreateContext(display, config, None, version)
            if (
                not surface
                or not context
                or not egl.eglMakeCurrent(display, surface, surface, context)
            ):
                raise RuntimeError("Cannot create a headless OpenGL 3.3 context")
            gl = ct.CDLL("libOpenGL.so.0")
            gl.glGetString.argtypes = [ct.c_uint]
            gl.glGetString.restype = ct.c_char_p
            result = {}
            for key, identity in (("vendor", 0x1F00), ("renderer", 0x1F01), ("version", 0x1F02)):
                value = gl.glGetString(identity)
                if not value:
                    raise RuntimeError("OpenGL context did not report " + key)
                result[key] = value.decode("utf-8", errors="replace")
            return result
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            egl.eglMakeCurrent(display, None, None, None)
            if context:
                egl.eglDestroyContext(display, context)
            if surface:
                egl.eglDestroySurface(display, surface)
            egl.eglTerminate(display)
    raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    try:
        print(json.dumps(probe()))
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)

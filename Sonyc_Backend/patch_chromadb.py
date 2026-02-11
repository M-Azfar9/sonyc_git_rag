"""
Patch chromadb config.py for pydantic v2 compatibility.
Run this script after installing/upgrading chromadb:
    python patch_chromadb.py
"""
import site
import os

site_packages = site.getsitepackages()[0]
config_path = os.path.join(site_packages, "chromadb", "config.py")

if not os.path.exists(config_path):
    print(f"chromadb config.py not found at {config_path}")
    exit(1)

with open(config_path, "r") as f:
    content = f.read()

patched = False

# Patch 1: Use pydantic_settings instead of pydantic.v1 fallback
old_import = """in_pydantic_v2 = False
try:
    from pydantic import BaseSettings
except ImportError:
    in_pydantic_v2 = True
    from pydantic.v1 import BaseSettings
    from pydantic.v1 import validator

if not in_pydantic_v2:
    from pydantic import validator  # type: ignore # noqa"""

new_import = """in_pydantic_v2 = False
try:
    from pydantic_settings import BaseSettings
    in_pydantic_v2 = True
    from pydantic import validator  # type: ignore # noqa
except ImportError:
    try:
        from pydantic import BaseSettings
        from pydantic import validator  # type: ignore # noqa
    except ImportError:
        in_pydantic_v2 = True
        from pydantic.v1 import BaseSettings
        from pydantic.v1 import validator"""

if old_import in content:
    content = content.replace(old_import, new_import)
    patched = True
    print("Patched: import block")

# Patch 2: Add type annotation to chroma_coordinator_host
if "    chroma_coordinator_host = " in content and "    chroma_coordinator_host: str = " not in content:
    content = content.replace(
        '    chroma_coordinator_host = "localhost"',
        '    chroma_coordinator_host: str = "localhost"'
    )
    patched = True
    print("Patched: chroma_coordinator_host type annotation")

# Patch 3: Add extra = "allow" to Config class
if 'extra = "allow"' not in content and "class Config:" in content:
    content = content.replace(
        '        env_file_encoding = "utf-8"',
        '        env_file_encoding = "utf-8"\n        extra = "allow"'
    )
    patched = True
    print("Patched: Config.extra = allow")

if patched:
    with open(config_path, "w") as f:
        f.write(content)
    print(f"Successfully patched {config_path}")
else:
    print("No patches needed (already patched or different version)")

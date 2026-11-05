#!/bin/sh
# Run a script with FreeCAD's headless Python: cad/fc.sh cad/build_part.py
FC=${FREECADCMD:-/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd}
exec "$FC" -c "import runpy, sys; sys.argv = ['$1']; runpy.run_path('$1', run_name='__main__')"

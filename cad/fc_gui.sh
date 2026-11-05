#!/bin/sh
# Run a FreeCAD script that needs the GUI (TechDraw page layout/export): cad/fc_gui.sh cad/make_drawings.py
# The script must end with os._exit(0). Gives up after ${FC_TIMEOUT:-300} s.
FC=${FREECAD_GUI:-/Applications/FreeCAD.app/Contents/MacOS/FreeCAD}
export BRACKET_CAD_DIR="$(cd "$(dirname "$0")" && pwd)"
"$FC" "$1" > "${BRACKET_CAD_DIR}/out/fc_gui.out" 2>&1 &
pid=$!
i=0
while kill -0 $pid 2>/dev/null; do
  sleep 1; i=$((i + 1))
  if [ $i -ge ${FC_TIMEOUT:-300} ]; then kill $pid; echo "timeout"; exit 1; fi
done
echo "finished in ${i} s"

#!/usr/bin/env bash

SRC=$(find "include" "examples" "units" "checks" "bench" "tools" \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cc" \))

for f in $SRC; do
	echo -e "- apply clang-format to file $f"
	clang-format -i $f
done

echo -e "--- clang-format DONE ---"

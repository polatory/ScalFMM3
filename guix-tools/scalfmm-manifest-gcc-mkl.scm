;; What follows is a "manifest" equivalent to the command line you gave.
;; You can store it in a file that you may then pass to any 'guix' command
;; that accepts a '--manifest' (or '-m') option.

(specifications->manifest
  (list "gcc-toolchain@11"
        "cmake"
        "make"
        "coreutils"
        "ncurses"
        "intel-oneapi-mkl"
        "grep"
        "findutils"
	"openmpi"
	"bash"
	"git"
	"python"
	"python-colorama"
	"inetutils"
        "sed"))

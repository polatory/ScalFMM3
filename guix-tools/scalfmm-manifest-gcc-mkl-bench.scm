;; What follows is a "manifest" equivalent to the command line you gave.
;; You can store it in a file that you may then pass to any 'guix' command
;; that accepts a '--manifest' (or '-m') option.

(specifications->manifest
  (list "bash"
        "cmake"
        "coreutils"
        "curl"
        "findutils"
        "gcc-toolchain@11"
        "git"
        "grep"
        "inetutils"
        "intel-oneapi-mkl"
        "jq"
        "jube-with-yaml"
        "make"
        "ncurses"
        "nss-certs"
	"python"
        "python-certifi"
        "python-click"
	"python-colorama"
        "python-elasticsearch"
        "python-matplotlib"
        "python-pandas"
        "python-pyyaml"
        "python-setuptools"
        "sed"
        "slurm"
        "sqlite"))


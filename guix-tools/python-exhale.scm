(define-module (python-exhale)
  #:use-module (guix)
  #:use-module (guix packages)
  #:use-module (guix download)
  #:use-module (guix git-download)
  #:use-module (guix hg-download)
  #:use-module (guix gexp)
  #:use-module (guix utils)
  #:use-module (guix build-system python)
  #:use-module (guix build-system pyproject)
  #:use-module ((guix licenses) #:prefix license:)
  #:use-module (gnu packages)
  #:use-module (gnu packages certs)
  #:use-module (gnu packages check)
  #:use-module (gnu packages fonts)
  #:use-module (gnu packages fontutils)
  #:use-module (gnu packages graphviz)
  #:use-module (gnu packages image)
  #:use-module (gnu packages imagemagick)
  #:use-module (gnu packages jupyter)
  #:use-module (gnu packages python)
  #:use-module (gnu packages sphinx)
  #:use-module (gnu packages xml)
  #:use-module (gnu packages python-build)
  #:use-module (gnu packages python-check)
  #:use-module (gnu packages python-crypto)
  #:use-module (gnu packages python-web)
  #:use-module (gnu packages python-xyz)
  #:use-module (gnu packages time)
  #:use-module (gnu packages python-science)
  #:use-module (gnu packages graph)
  #:use-module ((guix licenses) #:prefix license:)
  #:use-module (gnu packages)
  #:use-module (guix build-system gnu))

(define-public python-exhale
  (package
    (name "python-exhale")
    (version "0.3.7")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "exhale" version))
       (sha256
        (base32 "1n5hsrg7swh535bd5b3f55ldcb343yld849kjcfm2mlllp89cakm"))))
    (build-system pyproject-build-system)
    (propagated-inputs (list python-beautifulsoup4 python-breathe python-lxml
                             python-six python-sphinx))
    (native-inputs (list python-setuptools python-wheel))
    (home-page "https://github.com/svenevs/exhale")
    (synopsis
     "Automatic C++ library API documentation generator using Doxygen, Sphinx, and")
    (description
     "Automatic C++ library API documentation generator using Doxygen, Sphinx, and.")
    (license #f)))

;; python-exhale

# Bug Algorithms using Neural Networks
## ECE 780 To 8 Course Project by Andreas Stöckel

![Implementation of the Bug 0 algorithm as Nengo network](https://rawgithub.com/astoeckel/ece_780_bug_algorithms_nengo/master/doc/media/nengo_bug_0_network.svg)

This repository contains an implementation of the Bug 0 and 2 algorithms
proposed by Lumelsky and Stepanov in 1987 as a Nengo spiking neural network.

* [Project Report](https://github.com/astoeckel/ece_780_bug_algorithms_nengo/raw/master/doc/2017_08_ece_780_project.pdf) (PDF, 0.7 MB)
* [Presentation, no videos](https://github.com/astoeckel/ece_780_bug_algorithms_nengo/raw/master/doc/presentation/2017_07_24_ece_780_astoeckel_project.pdf) (PDF, 1.1 MB)
* [Presentation, with videos](https://somweyr.de/uni/2017_07_24_ece_780_astoeckel_project.html) (HTML, 26 MB)

## Requirements

This code requires Python 3, as well as the packages `numpy`, `scipy`,
`matplotlib` and `nengo`. You can use `nengo_gui` to graphically inspect
the neural network models (load the `gui.py` script with `nengo_gui`).

## How to run

To run simulations, use the following command
```
./simulation.py --map <MAP FILE NAME> --agent <AGENT CLASS NAME> --T <RUNTIME>
```
where `<MAP FILE NAME>` points at the map that should be used (e.g. from the
maps subfolder) and `<AGENT CLASS NAME>` is the Python agent class name as
defined in one of the files in the `agents` subdirectory.

To visualize use `visualize.py` and point it at a number of the `.pcl` files
produced by `simulation.py`. The `--animate` feature to produce videos requires
an installation of `ffmpeg` and `libx264`.

## Building the presentation

The presentation includes some videos produced by the `visualize.py` script.
Please use the included `pdf_to_html_presentation.sh` script to convert the
compiled LaTeX Beamer PDF to interactive HTML. The script requires the
programs `qpdf`, `pdfinfo`, as well as `pdf2svg`.

## License

The code is licensed under GPLv3, all documentation (the presentation and the
project report) are licensed CC BY SA.


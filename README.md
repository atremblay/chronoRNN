# chronoRNN

This semester project was part of the [ICLR 2018 Reproducibility Challenge](https://ift6135h18.wordpress.com/project-description-iclr2018-reproducibility-challenge/) where graduate students are asked to reproduce, if possible, the results of published papers. Our team chose to work on "[Can Recurrent Neural Networks Warp Time?](https://openreview.net/pdf?id=SJcKhk-Ab)"

Team Members (in alphabetical order)
- [Marc-Antoine Bélanger](https://github.com/gbmarc1)
- [Jules Gagnon-Marchand](https://github.com/julesgm)
- [Andrés Morales](https://github.com/jamorafo)
- [Alexis Tremblay](https://github.com/atremblay)

## Tests
Just run
```bash
python tests.py
```

## Tasks

### Copy task
Here are the commands to use to run the different copy tasks

*Std LSTM - Variable Copy - 500*
```
$ mkdir -p results/variable_copy_std_500
$ python main.py --task copyTask -pnum_batches=30000 --checkpoint_path results/variable_copy_std_500  -pmodel_type=LSTM -pbatch_size=32 -pchrono=0 -pvariable=True -prmsprop_lr=0.0001 -prmsprop_momentum=0
```

*Chrono LSTM - Variable Copy - 500*
```
$ mkdir -p results/variable_copy_chrono_500
$ python main.py --task copyTask -pnum_batches=30000 --checkpoint_path results/variable_copy_chrono_500  -pmodel_type=LSTM -pbatch_size=32 -pchrono=500 -pvariable=True -prmsprop_lr=0.0001 -prmsprop_momentum=0
```

*Std LSTM - Variable Copy - 1000*
```
$ mkdir -p results/variable_copy_std_1000
$ python main.py --task copyTask -pnum_batches=30000 --checkpoint_path results/variable_copy_std_1000  -pmodel_type=LSTM -pbatch_size=32 -pchrono=0 -pvariable=True -prmsprop_lr=0.0001 -prmsprop_momentum=0 -pseq_len=1000
```

*Chrono LSTM - Variable Copy - 1000*
```
$ mkdir -p results/variable_copy_chrono_1000
$ python main.py --task copyTask -pnum_batches=30000 --checkpoint_path results/variable_copy_chrono_1000  -pmodel_type=LSTM -pbatch_size=32 -pchrono=1000 -pvariable=True -prmsprop_lr=0.0001 -prmsprop_momentum=0 -pseq_len=1000
```

*Std LSTM - Copy - 500*
```
$ mkdir -p results/copy_std_500
$ python main.py --task copyTask -pnum_batches=30000 --checkpoint_path results/copy_std_500  -pmodel_type=LSTM -pbatch_size=32 -pchrono=0 -prmsprop_lr=0.0001 -prmsprop_momentum=0
```

*Chrono LSTM - Copy - 500*
```
$ mkdir -p results/copy_chrono_500
$ python main.py --task copyTask -pnum_batches=30000 --checkpoint_path results/copy_chrono_500  -pmodel_type=LSTM -pbatch_size=32 -pchrono=750 -prmsprop_lr=0.0001 -prmsprop_momentum=0
```

*Std LSTM - Copy - 2000*
```
$ mkdir -p results/copy_std_2000
$ python main.py --task copyTask -pnum_batches=30000 --checkpoint_path results/copy_std_2000  -pmodel_type=LSTM -pbatch_size=32 -pchrono=0 -prmsprop_lr=0.0001 -prmsprop_momentum=0 -pseq_len=2000
```

*Chrono LSTM - Copy - 2000*
```
$ mkdir -p results/copy_chrono_2000
$ python main.py --task copyTask -pnum_batches=30000 --checkpoint_path results/copy_chrono_2000  -pmodel_type=LSTM -pbatch_size=32 -pchrono=3000 -prmsprop_lr=0.0001 -prmsprop_momentum=0 -pseq_len=2000
```

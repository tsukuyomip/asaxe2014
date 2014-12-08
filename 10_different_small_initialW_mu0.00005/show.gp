set xlabel "t(Epochs)"
set ylabel "input-output mode strength"
plot [0:600]"experiment_result.dat" w l lw 1, "theory_result.dat" w l lw 3
set ter pos enh col eps
set o "my_fig3.eps"
rep
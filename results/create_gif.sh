/usr/bin/convert                                         \
  -delay 10                                              \
   $(for i in $(seq 0 1 29); do echo live_inf/task=6_step=${i}.png; done) \
   $(for i in $(seq 0 1 29); do echo live_inf/task=7_step=${i}.png; done) \
   $(for i in $(seq 0 1 29); do echo live_inf/task=8_step=${i}.png; done) \
   $(for i in $(seq 0 1 28); do echo live_inf/task=9_step=${i}.png; done) \
  -loop 0                                                \
  -delay 500                                             \
  live_inf/task=9_step=29.png                            \
   animated.gif

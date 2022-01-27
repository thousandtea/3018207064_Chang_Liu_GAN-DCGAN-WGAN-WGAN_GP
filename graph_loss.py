import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

# 读取文件数据
fp = open('loss/gan_loss.csv', 'r')

# 更新次数 d_loss g_loss
train_steps = []
d_loss = []
g_loss = []

# 解析数据
for line in fp.readlines():
    con = line.strip('\n').split(',')
    print(con)
    train_steps.append(int(con[0]))
    d_loss.append(float(con[1]))
    g_loss.append(float(con[2]))

# 绘制曲线图
host = host_subplot(111)
plt.subplots_adjust(right=0.8)
par1 = host.twinx()

# 设置类标
host.set_xlabel("steps")
host.set_ylabel("discriminator_loss")
par1.set_ylabel("generator_loss")

# 绘制曲线
p1, = host.plot(train_steps, d_loss, "b-", label="d_loss")
p2, = par1.plot(train_steps, g_loss, "r-", label="g_loss")


host.legend(loc=5)

# 设置颜色
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())


plt.draw()
plt.savefig('loss/gan_loss.png')

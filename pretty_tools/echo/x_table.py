from rich.console import Console
from rich.table import Table


class X_Table(Table):
    # todo 给出一些使用示例，免得每次都不知道咋用
    """生成一个表格，并设置高亮

    .. code-block:: python

        table = X_Table(title="exp args merge", highlight=True)

        #* 设置表头
        table.add_column("key", justify="left")
        table.add_column("value", justify="center")
        table.add_column("last_set", justify="center")

        # 添加数据
        table.add_row(str(k), str(v1), str(v2))
        ...

        # 打印表格
        table.print()

    .. image:: http://pb.x-contion.top/wiki/2023_08/25/3_x_table%E4%BD%BF%E7%94%A8%E7%A4%BA%E4%BE%8B.png
        :width: 600px
        :align: center
    """
    console = Console()

    def print(self):
        # self.console = Console()
        self.console.print(self)

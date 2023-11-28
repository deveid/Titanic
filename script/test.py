import pandas as pd

class model():
  def __init__(self):
    self.d = {'col1': [1, 2], 'col2': [3, 4]}
  
  def get_rec(self):
    df = pd.DataFrame(data=self.d)
    df.to_markdown('fer.md',index=False)

if __name__ == '__main__':
  model.get_rec()

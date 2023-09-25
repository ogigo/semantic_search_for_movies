import faiss
import numpy as np
import model
import data

encoded_data = model.model.encode(data.df.Plot.tolist())
encoded_data = np.asarray(encoded_data.astype('float32'))
index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
index.add_with_ids(encoded_data, np.array(range(0, len(data.df))))
faiss.write_index(index, 'movie_plot.index')
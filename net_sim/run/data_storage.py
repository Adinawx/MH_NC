import numpy as np
import os

class DataStorage:
    def __init__(self):
        self.data = {}

    def save_data(self, r, ch, trans_times, arrival_times, semi_dec_times, dec_times, ct_type_hist, hist_erasures, eps_mean_hist):
        key = (r, ch)
        self.data[key] = {
            'trans_times': trans_times,
            'arrival_times': arrival_times,
            'semi_dec_times': semi_dec_times,
            'dec_times': dec_times,
            'ct_type_hist': ct_type_hist,
            'hist_erasures': hist_erasures,
            'eps_mean_hist': eps_mean_hist
        }

    def save_to_files(self, res_folder):

        # Create the results folder
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        for (r, ch), data in self.data.items():
            np.save(os.path.join(res_folder, f"trans_times_rep={r}_ch={ch}.npy"), data['trans_times'])
            np.save(os.path.join(res_folder, f"arrival_times_rep={r}_ch={ch}.npy"), data['arrival_times'])
            np.save(os.path.join(res_folder, f"semi_dec_times_rep={r}_ch={ch}.npy"), data['semi_dec_times'])
            np.save(os.path.join(res_folder, f"dec_times_rep={r}_ch={ch}.npy"), data['dec_times'])
            np.save(os.path.join(res_folder, f"trans_types_rep={r}_ch={ch}.npy"), data['ct_type_hist'])
            np.save(os.path.join(res_folder, f"erasures_rep={r}_ch={ch}.npy"), data['hist_erasures'])
            np.save(os.path.join(res_folder, f"eps_mean_rep={r}_ch={ch}.npy"), data['eps_mean_hist'])

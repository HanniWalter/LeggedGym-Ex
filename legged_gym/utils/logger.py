from collections import defaultdict
import numpy as np
import xlsxwriter

EXCEL_FILENAME = "/home/lupinjia/Documents/2025/genesis_test/20250724_go2_deploy_step_gait.xlsx"


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        # Plotting support was removed to avoid creating image files during play.
        return

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
        
class QuadLogger(Logger):
    def __init__(self, dt):
        super().__init__(dt)
    
    def set_header_of_xlsx(self):
        self.workbook = xlsxwriter.Workbook(EXCEL_FILENAME)
        self.worksheet = self.workbook.add_worksheet()
        label = list(self.state_log.keys())
        for i in range(len(label)):
            self.worksheet.write(0, i, label[i])
    
    def save_data_to_xlsx(self):
        '''
        save the data to a excel file
        '''
        # set header
        self.set_header_of_xlsx()
        # get the first key, to get the length of the data
        first_key = list(self.state_log.keys())[0]
        for row in range(len(self.state_log[first_key])):
            for col, key in enumerate(self.state_log.keys()):
                self.worksheet.write(1+row, col, self.state_log[key][row])
        self.workbook.close()
        print("xlsx file created and filled!")


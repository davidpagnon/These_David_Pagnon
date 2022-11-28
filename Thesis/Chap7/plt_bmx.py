import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

IK_folder = r'G:\Op2Ani\20200908_6cams_xsens_Voreppe\seq024_bike\opensim'


def read_mot_sto(file):
    if file.endswith('.sto'):
        skip = 18
    elif file.endswith('.mot'):
        skip = 10
    else:
        raise Exception('Wrong file extension')
    df_file = pd.read_csv(file, sep='\t', skiprows=skip)
    return df_file


b, a = butter(4/2, 10/(120/2), 'low', analog = False) 

front_knee = read_mot_sto(os.path.join(IK_folder, r'seq024_filt_bikepilot_0-439.mot'))['knee_angle_l'][280:420]
front_knee = filtfilt(b,a,front_knee)

left_arm = read_mot_sto(os.path.join(IK_folder, r'seq024_filt_bikepilot_0-439.mot'))['arm_flex_l'][280:420]
right_arm = read_mot_sto(os.path.join(IK_folder, r'seq024_filt_bikepilot_0-439.mot'))['arm_flex_r'][280:420]
left_arm = filtfilt(b,a,left_arm)
right_arm = filtfilt(b,a,right_arm)

frontWheel = read_mot_sto(os.path.join(IK_folder, r'seq024_filt_bike_0-439_BodyKinematics_pos_global.sto'))['frontWheel_Y'][280:420]
frontWheel = filtfilt(b,a,frontWheel)

handlebar = read_mot_sto(os.path.join(IK_folder, r'seq024_filt_bikepilot_0-439.mot'))['bikeframehandlebar_angle'][280:420]
handlebar = filtfilt(b,a,handlebar)

COM_pilot = read_mot_sto(os.path.join(IK_folder, r'seq024_filt_pilot_0-439_BodyKinematics_vel_global.sto'))['center_of_mass_X'][280:420]
COM_bike = read_mot_sto(os.path.join(IK_folder, r'seq024_filt_bike_0-439_BodyKinematics_vel_global.sto'))['center_of_mass_X'][280:420]
COM_pilot = filtfilt(b,a,COM_pilot)
COM_bike = filtfilt(b,a,COM_bike)

bikepilot = read_mot_sto(os.path.join(IK_folder, r'seq024_filt_bikepilot_0-439_BodyKinematics_pos_global.sto'))[280:420]
footX = bikepilot['toes_r_X']-bikepilot['crankset_X']-0.07
footY = bikepilot['toes_r_Y']-bikepilot['crankset_Y']-0.02
pedalX = bikepilot['pedal_r_X']-bikepilot['crankset_X']
pedalY = bikepilot['pedal_r_Y']-bikepilot['crankset_Y']
footX = filtfilt(b,a,footX)
footY = filtfilt(b,a,footY)
pedalX = filtfilt(b,a,pedalX)
pedalY = filtfilt(b,a,pedalY)

time = np.arange(0,len(front_knee)/120,1/120)


fig, ax = plt.subplots(3,3, gridspec_kw={'width_ratios':[0.1,0.45,0.45]}, figsize=(8,10))

ax[0,0].text(0.5, 0.5, "Pilot", size=12, rotation=90., ha="center", va="center")
ax[0,0].axis('off')

ax[0,1].plot(time, front_knee)
ax[0,1].spines['top'].set_visible(False)
ax[0,1].spines['right'].set_visible(False)
ax[0,1].tick_params(labelbottom=False)
ax[0,1].set_ylabel('Front knee flexion (deg)', fontweight="bold")

ax[0,2].plot(time, left_arm, label='Left')
ax[0,2].plot(time, right_arm, label='Right')
ax[0,2].spines['top'].set_visible(False)
ax[0,2].spines['right'].set_visible(False)
ax[0,2].tick_params(labelbottom=False)
ax[0,2].set_ylabel('Shoulder flexion (deg)', fontweight="bold")
ax[0,2].legend()

ax[1,0].text(0.5, 0.5, "Bike", size=12, rotation=90., ha="center", va="center")
ax[1,0].axis('off')

ax[1,1].plot(time, frontWheel)
ax[1,1].spines['top'].set_visible(False)
ax[1,1].spines['right'].set_visible(False)
ax[1,1].tick_params(labelbottom=False)
ax[1,1].set_ylabel('Front wheel elevation (m)', fontweight="bold")

ax[1,2].plot(time, handlebar)
ax[1,2].spines['top'].set_visible(False)
ax[1,2].spines['right'].set_visible(False)
ax[1,2].tick_params(labelbottom=False)
ax[1,2].set_ylabel('Handlebar rotation (deg)', fontweight="bold")

ax[2,0].text(0.5, 0.5, "Bike & Pilot", size=12, rotation=90., ha="center", va="center")
ax[2,0].axis('off')

ax[2,1].plot(time, COM_pilot, label='Pilot')
ax[2,1].plot(time, COM_bike, label='Bike')
ax[2,1].spines['top'].set_visible(False)
ax[2,1].spines['right'].set_visible(False)
ax[2,1].set_ylabel('Forward COM speed (m/s)', fontweight="bold")
ax[2,1].set_xlabel('Time (s)', fontweight="bold")
ax[2,1].legend()

ax[2,2].plot(footX, footY, label='Foot')
ax[2,2].plot(pedalX, pedalY, label='Pedal')
ax[2,2].spines['top'].set_visible(False)
ax[2,2].spines['right'].set_visible(False)
ax[2,2].set_ylabel('Segment excursion (m)', fontweight="bold")
ax[2,2].set_xlabel('Time (s)', fontweight="bold")
ax[2,2].set_xlim(-0.2,0.2)
ax[2,2].set_ylim(-0.2,0.2)
ax[2,2].legend()

fig.tight_layout()
fig.show()
fig.savefig(os.path.join(IK_folder, f'KPIs_BMX.png'), dpi=300)


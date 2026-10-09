import MDAnalysis as mda
from MDAnalysis.transformations import unwrap
import numpy as np
from scipy import optimize
from tqdm import tqdm

#-----------------------------------------#
# Read trajectory - GROMACS .xtc
#-----------------------------------------#

topology_file = "run.tpr"   # or "topol.tpr" — any file MDAnalysis accepts
trajectory_file = "traj.xtc"

print('Reading GROMACS trajectory...')
u = mda.Universe(topology_file, trajectory_file)

# Add on-the-fly PBC unwrapping so coordinates are continuous (no jumps)
u.trajectory.add_transformations(unwrap(u.atoms))

#-----------------------------------------#
# Inputs — edit these for your system
#-----------------------------------------#
loading    = 225   # number of molecules
print_step = 1    # ps — must match your .xtc output frequency
origin_len_ps = 120
num_origin = 10
num_H      = 6    # H atoms per molecule

origin_steps = int(origin_len_ps / print_step)
num_timesteps = len(u.trajectory)
max_origin = num_timesteps - origin_steps - 1

# Box lengths (Angstrom) — MDAnalysis gives these per-frame but for a
# rigid box just read from frame 0
u.trajectory[0]
box = u.dimensions          # [lx, ly, lz, alpha, beta, gamma]
x_box, y_box, z_box = box[0], box[1], box[2]

#-----------------------------------------#
# Select hydrogen atoms and build H_list
#
# Adjust the selection strings to match YOUR atom/residue names.
# MDAnalysis selection syntax: "name H1", "type H", "resname MOL and name H*"
# Check your atom names with:  print(u.select_atoms("all").names)
#-----------------------------------------#

sel_HC1 = u.select_atoms("resname ETOH and name HC1")  # edit as needed
sel_HC2 = u.select_atoms("resname ETOH and name HC2")
sel_HC3 = u.select_atoms("resname ETOH and name HC3")
sel_HC4 = u.select_atoms("resname ETOH and name HC4")
sel_HC5 = u.select_atoms("resname ETOH and name HC5")
sel_HO  = u.select_atoms("resname ETOH and name HO")


selections = [sel_HC1, sel_HC2, sel_HC3, sel_HC4,
              sel_HC5,  sel_HO]

# Pre-load all coordinates into memory as arrays shaped (n_molecules, n_frames, 3)
# Same logical structure as the original H_list, just indexed differently below.
print("Loading coordinates into memory...")
H_list = []
for sel in selections:
    # shape: (n_frames, n_atoms_in_sel, 3)
    coords_over_time = np.array([sel.positions.copy() for ts in u.trajectory])
    # Rearrange to (n_molecules * n_frames, 3) to match original indexing:
    # H_i[molecule + loading * timestep]
    # coords_over_time shape is (n_frames, loading, 3)
    # We want H_i[molecule + loading*t] = coords_over_time[t, molecule, :]
    # So flatten to (n_frames * loading, 3) with frame as the slow index:
    n_frames = coords_over_time.shape[0]
    coords_flat = coords_over_time.reshape(n_frames * loading, 3)
    H_list.append(coords_flat)

print("Coordinates loaded.")

# ---- Everything below this line is UNCHANGED from your original script ----

#-----------------------------------------#
# Calculate ISF
#-----------------------------------------#

Q = [0.243538, 0.297811, 0.351816, 0.405524, 0.458890, 0.693626, 0.794381, 0.892463, 0.987518, 1.079253, 1.167332, 1.251484, 1.350691, 1.442777, 1.510990]
F_QT_list_Qsplit = []

print('Calculating ISF...')
for qi in tqdm(Q):
    F_QT_int_list = []
    origins = np.linspace(0, max_origin, num_origin, dtype=int)
    for origin in origins:
        F_QT_origin_list = [1]
        if origin == 0:
            for t in range(1, num_timesteps):
                F_QT = 0
                for molecule in range(loading):
                    for H_i in H_list:
                        sqr_dist = 0
                        for xyz in range(3):
                            if xyz == 0:
                                if H_i[molecule + loading*(origin+t-1)][xyz] - H_i[molecule + loading*(origin+t)][xyz] > x_box/2:
                                    H_i[molecule + loading*(origin+t)][xyz] += x_box
                                elif H_i[molecule + loading*(origin+t-1)][xyz] - H_i[molecule + loading*(origin+t)][xyz] <= -x_box/2:
                                    H_i[molecule + loading*(origin+t)][xyz] -= x_box
                                x_SD = (H_i[molecule + loading*origin][xyz] - H_i[molecule + loading*(origin+t)][xyz])**2
                            elif xyz == 1:
                                if H_i[molecule + loading*(origin+t-1)][xyz] - H_i[molecule + loading*(origin+t)][xyz] > y_box/2:
                                    H_i[molecule + loading*(origin+t)][xyz] += y_box
                                elif H_i[molecule + loading*(origin+t-1)][xyz] - H_i[molecule + loading*(origin+t)][xyz] <= -y_box/2:
                                    H_i[molecule + loading*(origin+t)][xyz] -= y_box
                                y_SD = (H_i[molecule + loading*origin][xyz] - H_i[molecule + loading*(origin+t)][xyz])**2
                            elif xyz == 2:
                                if H_i[molecule + loading*(origin+t-1)][xyz] - H_i[molecule + loading*(origin+t)][xyz] > z_box/2:
                                    H_i[molecule + loading*(origin+t)][xyz] += z_box
                                elif H_i[molecule + loading*(origin+t-1)][xyz] - H_i[molecule + loading*(origin+t)][xyz] <= -z_box/2:
                                    H_i[molecule + loading*(origin+t)][xyz] -= z_box
                                z_SD = (H_i[molecule + loading*origin][xyz] - H_i[molecule + loading*(origin+t)][xyz])**2
                        sqr_dist += x_SD + y_SD + z_SD
                        F_QT += (np.sin(qi * np.sqrt(sqr_dist))) / (qi * np.sqrt(sqr_dist))
                F_QT_avg = F_QT / (loading * num_H)
                if t <= origin_steps:
                    F_QT_origin_list.append(F_QT_avg)
        else:
            for t in range(1, origin_steps):
                F_QT = 0
                for molecule in range(loading):
                    for H_i in H_list:
                        x_SD = (H_i[molecule + loading*origin][0] - H_i[molecule + loading*(origin+t)][0])**2
                        y_SD = (H_i[molecule + loading*origin][1] - H_i[molecule + loading*(origin+t)][1])**2
                        z_SD = (H_i[molecule + loading*origin][2] - H_i[molecule + loading*(origin+t)][2])**2
                        sqr_dist = x_SD + y_SD + z_SD
                        F_QT += (np.sin(qi * np.sqrt(sqr_dist))) / (qi * np.sqrt(sqr_dist))
                F_QT_avg = F_QT / (loading * num_H)
                F_QT_origin_list.append(F_QT_avg)
        F_QT_int_list.append(F_QT_origin_list)

    F_QT_list = []
    for t in range(origin_steps):
        F_QT_val = sum(origin[t] for origin in F_QT_int_list) / len(F_QT_int_list)
        F_QT_list.append(F_QT_val)
    F_QT_list_Qsplit.append(F_QT_list)

#-----------------------------------------#
#Write ISF Output#
#-----------------------------------------# 

output = "self-part of the intermediate scattering function, first row is Q values, \n"
output += " timestep, 0.243538, 0.297811, 0.351816, 0.405524, 0.458890, 0.693626, 0.794381, 0.892463, 0.987518, 1.079253, 1.167332, 1.251484, 1.350691, 1.442777, 1.510990, \n"
timestep = 0
while timestep < origin_steps:
    Q_value = 0
    output += str(timestep) + ","
    while Q_value < len(Q): 
        output += str(F_QT_list_Qsplit[Q_value][timestep]) + ","
        Q_value += 1
    output += "\n"
    timestep += 1  

with open("ISF.csv","w") as OutputFile:
    OutputFile.write(output)

#-----------------------------------------#
#EISF fitting#
#-----------------------------------------# 
print("Fitting ISF functions...")

def func_1(FQT,pre,expo,base,):
    return pre*(np.exp(-expo*FQT)) + base
    
def func_2(FQT, Base, ratio, expo, expo_2):
    pre_total = 1 - Base
    pre = pre_total * ratio
    pre_2 = pre_total * (1-ratio)
    return pre*(np.exp(-expo*FQT)) + pre_2*(np.exp(-expo_2*FQT)) + Base


params_list = []
guess = [(F_QT_list_Qsplit[0][-1]-0.02),1,0.05,1.5]
for q_val in F_QT_list_Qsplit:
    xdata = np.arange(len(q_val))*print_step
    ydata = q_val
    guess[0]=ydata[-1]-0.02
    params, params_covariance = optimize.curve_fit(func_2, xdata, ydata, bounds=([0,0,0,0],[ydata[-1],1,20,20]),p0=guess)  
    guess = [0]
    guess.extend(params[1:4])
    params_list.append(params)


#-----------------------------------------#
#write fitting output files#
#-----------------------------------------# 


output_fit = " timestep, 0.243538, 0.297811, 0.351816, 0.405524, 0.458890, 0.693626, 0.794381, 0.892463, 0.987518, 1.079253, 1.167332, 1.251484, 1.350691, 1.442777, 1.510990, \n"
timestep = 0
while timestep < origin_steps:
    Q_value = 0
    output_fit += str(timestep) + ","
    while Q_value < len(Q): 
        output_fit += str(func_2((timestep*print_step),params_list[Q_value][0],params_list[Q_value][1],params_list[Q_value][2],params_list[Q_value][3])) + ","
        Q_value += 1
    output_fit += "\n"
    timestep += 1 

 
with open("ISF_fit.csv","w") as OutputFile:
    OutputFile.write(output_fit)
output_2 = "params, \n"
output_2 += "Q value, pre-exponential, exponent, baseline, \n"


output_1 = "Q value, pre-exponential 1, exponent 1, pre-exponential 2, exponent 2, Baseline \n"
for i, q in enumerate(params_list):
    output_1 += str(Q[i]) + "," + str((1-q[0])*q[1]) + "," + str(q[2])+ "," + str((1-q[0])*(1-q[1])) + "," + str(q[3]) + "," + str(q[0]) + "\n" 


with open("ISF_params.csv","w") as OutputFile:
    OutputFile.write(output_1)    

print("Complete.")

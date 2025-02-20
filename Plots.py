
import pandas as pd 
import matplotlib.pyplot as plt  # Corrected import
import numpy as np


ATE_DATA_Col = pd.read_csv(r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\kernel_conditions_Colangelo.csv")

ATE_DATA_JC = pd.read_csv(r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\kernel_conditions_Job_Corps.csv")

CATE_DATA_JC = pd.read_csv(r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\kernel_conditions_Job_Corps_CATE.csv")

CATE_DATA_syn = pd.read_csv(r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\kernel_conditions_Synthetic_CATE.csv")

CATE_DATA_syn_n = pd.read_csv(r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\kernel_conditions_Synthetic_Scaling_CATE.csv")

CATE_DATA_syn_n_const = pd.read_csv(r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\kernel_conditions_Synthetic_Scaling_CATE_constlam.csv")


CATE_DATA_JC_n = pd.read_csv(r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\kernel_conditions_JC_Scaling_CATE.csv")

CATE_DATA_JC_n_const = pd.read_csv(r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\kernel_conditions_JC_Scaling_CATE_const_lamb.csv")



path = r"C:\Users\Rishi\OneDrive\Documents\GitHub\Causal-Data-Checks\Plots"



# # ATE dataset

# # Set the font size for the plots
# plt.rcParams.update({'font.size': 16})

# # Create the combined plot for ATE_DATA_Col and ATE_DATA_JC
# plt.figure(figsize=(11.7, 4.3))  # A4 size in inches
# # Plot ATE_DATA_Col
# plt.plot(ATE_DATA_Col.iloc[:, 0], ATE_DATA_Col.iloc[:, 1], label='Colangelo Dataset', color='b')

# # Plot ATE_DATA_JC
# plt.plot(ATE_DATA_JC.iloc[:, 0], ATE_DATA_JC.iloc[:, 2], label='Job Corp Dataset', color='g')

# # Add labels and legend
# plt.xlabel('Lambda (Regularisation)', fontsize=16)
# plt.ylabel('Condition Number', fontsize=16)
# plt.legend()

# plt.tight_layout()


# # Specify the filename
# filename = "ate_combined_plot.pdf"

# # Save the plot to the specified path with the filename
# plt.savefig(f"{path}/{filename}",format = "pdf")


# # # Show the plot
# # plt.show()






# #const lam data

# # Set the font size for the plots
# plt.rcParams.update({'font.size': 16})

# # Create the plot with column 1 as the x-axis, and columns 3 and 5 as y-axis
# plt.figure(figsize=(11.7, 4.3))  # A4 size in inches

# # Plot column 3
# plt.plot(CATE_DATA_syn_n_const.iloc[:, 0], CATE_DATA_syn_n_const.iloc[:, 2], label='A1', color='b')

# # Plot column 5
# plt.plot(CATE_DATA_syn_n_const.iloc[:, 0], CATE_DATA_syn_n_const.iloc[:, 4], label='A2', color='g')

# # Add labels and legend
# plt.xlabel('n (datasize)', fontsize=16)
# plt.ylabel('Condition Number', fontsize=16)
# plt.legend()

# plt.tight_layout()

# # Specify the filename
# filename = "constant_lambda.pdf"

# # Save the plot to the specified path with the filename
# plt.savefig(f"{path}/{filename}",format = "pdf")


# # # Show the plot
# # plt.show()












#const lam data

# Set the font size for the plots
plt.rcParams.update({'font.size': 16})

# Create the plot with column 1 as the x-axis, and columns 3 and 5 as y-axis
plt.figure(figsize=(11.7, 4.3))  # A4 size in inches

# Plot column 3
plt.plot(CATE_DATA_JC_n_const.iloc[:, 0], CATE_DATA_JC_n_const.iloc[:, 2], label='A1', color='b')

# Plot column 5
plt.plot(CATE_DATA_JC_n_const.iloc[:, 0], CATE_DATA_JC_n_const.iloc[:, 4], label='A2', color='g')

# Add labels and legend
plt.xlabel('n (datasize)', fontsize=16)
plt.ylabel('Condition Number', fontsize=16)
plt.legend()

plt.tight_layout()

# Specify the filename
filename = "JC_constant_lambda.pdf"

# Save the plot to the specified path with the filename
plt.savefig(f"{path}/{filename}",format = "pdf")


# # Show the plot
# plt.show()

























# #CATE data

# # Set the font size for the plots
# plt.rcParams.update({'font.size': 16})

# # Create the plot with column 1 as the x-axis, and columns 3 and 5 as y-axis
# plt.figure(figsize=(11.7, 4.3))  # A4 size in inches

# # Plot column 3
# plt.plot(CATE_DATA_JC.iloc[:, 0], CATE_DATA_JC.iloc[:, 1], label='A1', color='b')

# # Plot column 5
# plt.plot(CATE_DATA_JC.iloc[:, 0], CATE_DATA_JC.iloc[:, 2], label='A2', color='g')

# # Add labels and legend
# plt.xlabel('Lambda (Regularisation)', fontsize=16)
# plt.ylabel('Condition Number', fontsize=16)
# plt.legend()

# plt.tight_layout()

# # Specify the filename
# filename = "Cate_JC.pdf"

# # Save the plot to the specified path with the filename
# plt.savefig(f"{path}/{filename}",format = "pdf")

# # # Show the plot
# # plt.show()







# #CATE data

# # Set the font size for the plots
# plt.rcParams.update({'font.size': 16})

# # Create the plot with column 1 as the x-axis, and columns 3 and 5 as y-axis
# plt.figure(figsize=(11.7, 4.3))  # A4 size in inches

# # Plot column 3
# plt.plot(CATE_DATA_syn.iloc[:, 0], CATE_DATA_syn.iloc[:, 1], label='A1', color='b')

# # Plot column 5
# plt.plot(CATE_DATA_syn.iloc[:, 0], CATE_DATA_syn.iloc[:, 2], label='A2', color='g')

# # Add labels and legend
# plt.xlabel('Lambda (Regularisation)', fontsize=16)
# plt.ylabel('Condition Number', fontsize=16)
# plt.legend()

# plt.tight_layout()

# # Specify the filename
# filename = "Cate_syn.pdf"

# # Save the plot to the specified path with the filename
# plt.savefig(f"{path}/{filename}",format = "pdf")

# # # Show the plot
# # plt.show()







# #CATE data

# # Set the font size for the plots
# plt.rcParams.update({'font.size': 16})

# # Calculate a simple moving average
# window_size = 10  # Adjust this window size as needed
# A1_moving_avg = CATE_DATA_syn_n.iloc[:, 2].rolling(window=window_size).mean()
# lambda_moving_avg = CATE_DATA_syn_n.iloc[:, 1].rolling(window=window_size).mean()

# # Create the plot with column 1 as the x-axis, and columns 2 and 3 as y-axis
# fig, ax1 = plt.subplots(figsize=(11.7, 4.3))  # A4 size in inches

# # Plot column 2 (A1) on the primary y-axis with dotted line and translucent color
# ax1.plot(CATE_DATA_syn_n.iloc[:, 0], CATE_DATA_syn_n.iloc[:, 2], label='A1', color='b', linestyle='--', alpha=0.1)

# # Plot moving average for A1
# ax1.plot(CATE_DATA_syn_n.iloc[:, 0], A1_moving_avg, label='A1 Moving Average', color='b')

# # Set the primary y-axis labels and colors
# ax1.set_xlabel('n (Datasize)', fontsize=16)
# ax1.set_ylabel('Condition Number (A1)', fontsize=16, color='b')
# ax1.tick_params(axis='y', labelcolor='b')

# # Create a second y-axis for lambda on the right side
# ax2 = ax1.twinx()

# # Plot column 1 vs column 1 (lambda) on the secondary y-axis with dotted line and translucent color
# ax2.plot(CATE_DATA_syn_n.iloc[:, 0], CATE_DATA_syn_n.iloc[:, 1], label='Lambda', color='r', linestyle='--', alpha=0.1)

# # Plot moving average for lambda
# ax2.plot(CATE_DATA_syn_n.iloc[:, 0], lambda_moving_avg, label='Lambda Moving Average', color='r')

# # Set the secondary y-axis labels and colors
# ax2.set_ylabel('Lambda', fontsize=16, color='r')
# ax2.tick_params(axis='y', labelcolor='r')

# # Add legend for both plots
# ax1.legend(loc="upper left")
# ax2.legend(loc="upper right")

# plt.tight_layout()


# # Specify the filename
# filename = "Cate_n1.pdf"

# # Save the plot to the specified path with the filename
# plt.savefig(f"{path}/{filename}",format = "pdf")
# # Show the plot
# # plt.show()
















# #CATE data

# # Set the font size for the plots
# plt.rcParams.update({'font.size': 16})

# # Calculate a simple moving average
# window_size = 10  # Adjust this window size as needed
# A1_moving_avg = CATE_DATA_syn_n.iloc[:, 4].rolling(window=window_size).mean()
# lambda_moving_avg = CATE_DATA_syn_n.iloc[:, 3].rolling(window=window_size).mean()

# # Create the plot with column 1 as the x-axis, and columns 2 and 3 as y-axis
# fig, ax1 = plt.subplots(figsize=(11.7, 4.3))  # A4 size in inches

# # Plot column 2 (A1) on the primary y-axis with dotted line and translucent color
# ax1.plot(CATE_DATA_syn_n.iloc[:, 0], CATE_DATA_syn_n.iloc[:, 4], label='A2', color='b', linestyle='--', alpha=0.1)

# # Plot moving average for A1
# ax1.plot(CATE_DATA_syn_n.iloc[:, 0], A1_moving_avg, label='A2 Moving Average', color='b')

# # Set the primary y-axis labels and colors
# ax1.set_xlabel('n (Datasize)', fontsize=16)
# ax1.set_ylabel('Condition Number (A2)', fontsize=16, color='b')
# ax1.tick_params(axis='y', labelcolor='b')

# # Create a second y-axis for lambda on the right side
# ax2 = ax1.twinx()

# # Plot column 1 vs column 1 (lambda) on the secondary y-axis with dotted line and translucent color
# ax2.plot(CATE_DATA_syn_n.iloc[:, 0], CATE_DATA_syn_n.iloc[:, 3], label='Lambda', color='r', linestyle='--', alpha=0.1)

# # Plot moving average for lambda
# ax2.plot(CATE_DATA_syn_n.iloc[:, 0], lambda_moving_avg, label='Lambda Moving Average', color='r')

# # Set the secondary y-axis labels and colors
# ax2.set_ylabel('Lambda', fontsize=16, color='r')
# ax2.tick_params(axis='y', labelcolor='r')

# # Add legend for both plots
# ax1.legend(loc="upper left")
# ax2.legend(loc="upper right")

# plt.tight_layout()


# # Specify the filename
# filename = "Cate_n2.pdf"

# # Save the plot to the specified path with the filename
# plt.savefig(f"{path}/{filename}",format = "pdf")

# # Show the plot
# # plt.show()





















#CATE data JC scaling

# Set the font size for the plots
plt.rcParams.update({'font.size': 16})

# Calculate a simple moving average
window_size = 10  # Adjust this window size as needed
A1_moving_avg = CATE_DATA_JC_n.iloc[:, 2].rolling(window=window_size).mean()
lambda_moving_avg = CATE_DATA_JC_n.iloc[:, 1].rolling(window=window_size).mean()

# Create the plot with column 1 as the x-axis, and columns 2 and 3 as y-axis
fig, ax1 = plt.subplots(figsize=(11.7, 4.3))  # A4 size in inches

# Plot column 2 (A1) on the primary y-axis with dotted line and translucent color
ax1.plot(CATE_DATA_JC_n.iloc[:, 0], CATE_DATA_JC_n.iloc[:, 2], label='A1', color='b', linestyle='--', alpha=0.1)

# Plot moving average for A1
ax1.plot(CATE_DATA_JC_n.iloc[:, 0], A1_moving_avg, label='A1 Moving Average', color='b')

# Set the primary y-axis labels and colors
ax1.set_xlabel('n (Datasize)', fontsize=16)
ax1.set_ylabel('Condition Number (A1)', fontsize=16, color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for lambda on the right side
ax2 = ax1.twinx()

# Plot column 1 vs column 1 (lambda) on the secondary y-axis with dotted line and translucent color
ax2.plot(CATE_DATA_JC_n.iloc[:, 0], CATE_DATA_JC_n.iloc[:, 1], label='Lambda', color='r', linestyle='--', alpha=0.1)

# Plot moving average for lambda
ax2.plot(CATE_DATA_JC_n.iloc[:, 0], lambda_moving_avg, label='Lambda Moving Average', color='r')

# Set the secondary y-axis labels and colors
ax2.set_ylabel('Lambda', fontsize=16, color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add legend for both plots
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()


# Specify the filename
filename = "Cate_n1_JC.pdf"

# Save the plot to the specified path with the filename
plt.savefig(f"{path}/{filename}",format = "pdf")
# Show the plot
# plt.show()
















#CATE data

# Set the font size for the plots
plt.rcParams.update({'font.size': 16})

# Calculate a simple moving average
window_size = 10  # Adjust this window size as needed
A1_moving_avg = CATE_DATA_JC_n.iloc[:, 4].rolling(window=window_size).mean()
lambda_moving_avg = CATE_DATA_JC_n.iloc[:, 3].rolling(window=window_size).mean()

# Create the plot with column 1 as the x-axis, and columns 2 and 3 as y-axis
fig, ax1 = plt.subplots(figsize=(11.7, 4.3))  # A4 size in inches

# Plot column 2 (A1) on the primary y-axis with dotted line and translucent color
ax1.plot(CATE_DATA_JC_n.iloc[:, 0], CATE_DATA_JC_n.iloc[:, 4], label='A2', color='b', linestyle='--', alpha=0.1)

# Plot moving average for A1
ax1.plot(CATE_DATA_JC_n.iloc[:, 0], A1_moving_avg, label='A2 Moving Average', color='b')

# Set the primary y-axis labels and colors
ax1.set_xlabel('n (Datasize)', fontsize=16)
ax1.set_ylabel('Condition Number (A2)', fontsize=16, color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for lambda on the right side
ax2 = ax1.twinx()

# Plot column 1 vs column 1 (lambda) on the secondary y-axis with dotted line and translucent color
ax2.plot(CATE_DATA_JC_n.iloc[:, 0], CATE_DATA_JC_n.iloc[:, 3], label='Lambda', color='r', linestyle='--', alpha=0.1)

# Plot moving average for lambda
ax2.plot(CATE_DATA_JC_n.iloc[:, 0], lambda_moving_avg, label='Lambda Moving Average', color='r')

# Set the secondary y-axis labels and colors
ax2.set_ylabel('Lambda', fontsize=16, color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add legend for both plots
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()


# Specify the filename
filename = "Cate_n2_JC.pdf"

# Save the plot to the specified path with the filename
plt.savefig(f"{path}/{filename}",format = "pdf")

# Show the plot
# plt.show()



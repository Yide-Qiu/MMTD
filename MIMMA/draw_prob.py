import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
os.makedirs('figures', exist_ok=True)


association_prob_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ground_thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]


HOTA_sim = np.random.uniform(20, 60, len(association_prob_thresholds))
IDF1_sim = np.random.uniform(15, 35, len(association_prob_thresholds))
AssA_sim = np.random.uniform(60, 80, len(association_prob_thresholds))
DetA_sim = np.random.uniform(40, 60, len(association_prob_thresholds))

HOTA_otb = np.random.uniform(20, 60, len(ground_thresholds))
IDF1_otb = np.random.uniform(15, 35, len(ground_thresholds))
AssA_otb = np.random.uniform(60, 80, len(ground_thresholds))
DetA_otb = np.random.uniform(40, 60, len(ground_thresholds))
pdb.set_trace()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(association_prob_thresholds, HOTA_sim, 'r-', label='HOTA')
axes[0].plot(association_prob_thresholds, IDF1_sim, 'b--', label='IDF1')
axes[0].plot(association_prob_thresholds, AssA_sim, 'g-.', label='AssA')
axes[0].plot(association_prob_thresholds, DetA_sim, 'm:', label='DetA')
axes[0].set_xlabel('Association Probability Threshold')
axes[0].set_ylabel('Score')
axes[0].set_title('Sensitivity Analysis on Association Probability Threshold')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(ground_thresholds, HOTA_otb, 'r-', label='HOTA')
axes[1].plot(ground_thresholds, IDF1_otb, 'b--', label='IDF1')
axes[1].plot(ground_thresholds, AssA_otb, 'g-.', label='AssA')
axes[1].plot(ground_thresholds, DetA_otb, 'm:', label='DetA')
axes[1].set_xlabel('Ground Threshold')
axes[1].set_ylabel('Score')
axes[1].set_title('Sensitivity Analysis on Ground Threshold')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('figures/sensitivity_analysis.png', dpi=300)
plt.show()

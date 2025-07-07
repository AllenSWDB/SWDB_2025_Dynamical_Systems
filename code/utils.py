import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import io
import matplotlib.cm as cm

def create_frame(transformed_actor_hidden_states_training_set, transformed_actor_hidden_states_testing_set,
                 a1_probs_training_set, example_session_id, elev, azim, axes_set = [0,1,2], frame = -1):

    if frame > np.shape(transformed_actor_hidden_states_testing_set)[1]:
        frame = np.shape(transformed_actor_hidden_states_testing_set)[1]
        
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        transformed_actor_hidden_states_training_set[:, :, axes_set[0]],
        transformed_actor_hidden_states_training_set[:, :, axes_set[1]],
        transformed_actor_hidden_states_training_set[:, :, axes_set[2]],
        s=40.0,
        c=a1_probs_training_set, cmap=cm.coolwarm,
        vmin=0, vmax=1, alpha = .2
    )
    
    ax.plot(
        transformed_actor_hidden_states_testing_set[example_session_id, :frame, axes_set[0]],
        transformed_actor_hidden_states_testing_set[example_session_id, :frame, axes_set[1]],
        transformed_actor_hidden_states_testing_set[example_session_id, :frame, axes_set[2]],
        color='k', alpha = .5,linewidth = 3
    )

    for tw in range(5):
        frametail = np.max(frame - tw,0)
        ax.plot(
            transformed_actor_hidden_states_testing_set[example_session_id, frametail:frame+1, axes_set[0]],
            transformed_actor_hidden_states_testing_set[example_session_id, frametail:frame+1, axes_set[1]],
            transformed_actor_hidden_states_testing_set[example_session_id, frametail:frame+1, axes_set[2]],
            color='k', alpha = .5, linewidth = 10 - tw
        )
    
    ax.plot(
        transformed_actor_hidden_states_testing_set[example_session_id, frame, axes_set[0]],
        transformed_actor_hidden_states_testing_set[example_session_id, frame, axes_set[1]],
        transformed_actor_hidden_states_testing_set[example_session_id, frame, axes_set[2]],
        color='k', marker='o', ms=15, alpha = .8
    )    
    
    ax.set_xlabel('PC '+str(axes_set[0]))
    ax.set_ylabel('PC '+str(axes_set[1]))
    ax.set_zlabel('PC '+str(axes_set[2]))
    
    fig.colorbar(scatter, ax=ax, label='Probability of Licking Right')
    
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    
    fig.tight_layout()
    
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    plt.close(fig)
    
    return buf


def plot_FPs(ax, sorted_fps, eig_decomps, D_use, n_interp=20, stability_thresh=1, ms=20,
             plot_unstable=True, plot_expansion=False, rf=550):
    """
    Plots fixed points (FPs) and their stability characteristics on the given axis.

    Parameters:
    - ax: matplotlib axis object where the plot will be drawn.
    - sorted_fps: Array of sorted fixed points.
    - eig_decomps: List of eigenvalue decompositions for each fixed point.
    - D_use: Transformation matrix for plotting.
    - n_interp: Number of interpolation points (not used in this function but can be for future expansion).
    - stability_thresh: Threshold to determine if a fixed point is stable.
    - ms: Marker size for plotting fixed points.
    - plot_unstable: Boolean flag to plot unstable fixed points.
    - plot_expansion: Boolean flag to plot expansion lines for unstable fixed points.
    - rf: Scaling factor for expansion lines just to see them in the visualization
    """
    
    # Define color map and default color
    cmap_grad = plt.get_cmap('plasma')
    color_stable = 'k'
    color_unstable = 'w'

    for fp_index, fp in enumerate(sorted_fps):
        # Project fixed point into the 2D plot space
        projected_fp = np.dot(fp, D_use[:, [0, 1]])

        # Check stability based on eigenvalues
        max_real_eigval = np.max(np.abs(eig_decomps[fp_index]['evals'].real))

        if max_real_eigval > stability_thresh:
            # Plot expansion lines for unstable fixed points if requested
            if plot_expansion:
                unstable_indices = np.argwhere(eig_decomps[fp_index]['evals'] > 1) + 1
                if unstable_indices.size > 0:
                    for index in range(np.max(unstable_indices)):
                        # Compute (project) right and left eigenvectors 
                        right_dots = np.dot(np.real(eig_decomps[fp_index]['R'][:, index]).T, D_use[:, [0, 1]])
                        left_dots = np.dot(np.real(eig_decomps[fp_index]['L'][:, index]).T, D_use[:, [0, 1]])
                        
                        # This calculates how perturbations move through full relaxation dynamics
                        # Compute the overlap between right and left eigenvectors
                        overlap = np.dot(right_dots, left_dots.T)
                        print(overlap)
                        # Calculate the start and end points for the expansion line
                        start_points = projected_fp - rf * overlap * right_dots
                        end_points = projected_fp + rf * overlap * right_dots
                        expansion_points = np.concatenate((start_points, end_points), axis=0)

                        # Extract x and y coordinates for plotting
                        x_coords = expansion_points[0::2]  # Every even index (0, 2, ...)
                        y_coords = expansion_points[1::2]  # Every odd index (1, 3, ...)

                        # Plot the expansion line
                        ax.plot(x_coords, y_coords, color='k', alpha=0.7, linewidth=4)

            # Plot fixed point with different colors based on stability
            if plot_unstable:
                ax.plot(projected_fp[0], projected_fp[1], '*',
                        color=color_stable, linewidth=10, markersize=ms,
                        markerfacecolor='w', markeredgewidth=3)
        else:
            ax.plot(projected_fp[0], projected_fp[1], '*',
                    color=color_unstable, linewidth=10, markersize=ms,
                    markerfacecolor='k')


            
def comp_eig_decomp(Ms, sort_by='real',do_compute_lefts=True):
  """Compute the eigenvalues of the matrix M. No assumptions are made on M.

  Arguments: 
    M: 3D np.array nmatrices x dim x dim matrix
    do_compute_lefts: Compute the left eigenvectors? Requires a pseudo-inverse 
      call.

  Returns: 
    list of dictionaries with eigenvalues components: sorted 
      eigenvalues, sorted right eigenvectors, and sored left eigenvectors 
      (as column vectors).
  """
  if sort_by == 'magnitude':
    sort_fun = np.abs
  elif sort_by == 'real':
    sort_fun = np.real
  else:
    assert False, "Not implemented yet."      
  
  decomps = []
  L = None  
  for M in Ms:
    evals, R = np.linalg.eig(M)    
    indices = np.flipud(np.argsort(sort_fun(evals)))
    if do_compute_lefts:
      L = np.linalg.pinv(R).T  # as columns
    decomps.append({'evals' : evals[indices], 'R' : R[:, indices],  'L' : L[:, indices]})
  
  return decomps

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
import seaborn as sns
from scipy.signal import fftconvolve as fftconv
import skimage
from scipy.interpolate import interp1d
import scipy.optimize as opt
import visual_stimuli as vs

plt.rcParams['image.cmap'] = 'gray'


class ParasolMosaic(object):

    on_parasol_datafile = 'mosaicGLM_apricot_ONParasol-1-mat.mat'
    sampling_rate = 120  # sampled at 120 Hz, ie 1 frame = 8.333 ms
    tsteps_postspike_filter = 120
    tsteps_stimulusfilter = 15
    data_pix_per_deg = 2   # TODO - check correct value

    bad_indices = [15, 67, 71, 86, 89,             # Bad data
                   9, 18, 20, 24, 25, 31, 50, 55]  # Bad fits

    def __init__(self):
        raw_data = sio.loadmat(ParasolMosaic.on_parasol_datafile)
        self.data = raw_data['mosaicGLM'][0]

        self.n_cells = len(self.data)
        print('Loaded data with %d cells' % self.n_cells)

        self.rgc_spatial_params = pd.DataFrame(columns=['center_x', 'center_y', 'sd_x', 'sd_y', 'center_rot'])
        self.tonicdrive = np.zeros((self.n_cells, 1))
        self.postspike_filter = np.zeros((self.n_cells, ParasolMosaic.tsteps_postspike_filter))
        self.stimulus_filter = np.zeros((self.n_cells, 13, 13, ParasolMosaic.tsteps_stimulusfilter))

        self.extract_data_spatial_params()
        self.extract_data_filter_components()

    def extract_data_spatial_params(self):

        cell_rf_list = []  # rf = receptive field
        for cellnum in range(self.n_cells):
            cellpar = {
                       'center_x': self.data[cellnum][0][0][3][0][0][0][0][0],
                       'center_y': self.data[cellnum][0][0][3][0][0][1][0][0],
                       'sd_x':     self.data[cellnum][0][0][3][0][0][2][0][0],
                       'sd_y':     self.data[cellnum][0][0][3][0][0][3][0][0],
                       'center_rot': self.data[cellnum][0][0][3][0][0][4][0][0]}
            cell_rf_list.append(cellpar)

        self.rgc_spatial_params = pd.DataFrame(cell_rf_list)

    def extract_data_filter_components(self):

        self.tonicdrive = np.array([self.data[cellnum][0][0][0][0][0][1][0][0][0][0][0] for cellnum in range(self.n_cells)])
        self.postspike_filter = np.array([self.data[cellnum][0][0][0][0][0][2][0][0][0] for cellnum in range(self.n_cells)])
        self.stimulus_filter = np.array([self.data[cellnum][0][0][0][0][0][3][0][0][0] for cellnum in range(self.n_cells)])
        self.space_rk1 = np.array([self.data[cellnum][0][0][0][0][0][3][0][0][2] for cellnum in range(self.n_cells)])
        self.time_rk1 = np.array([self.data[cellnum][0][0][0][0][0][3][0][0][3] for cellnum in range(self.n_cells)])
        #self.time_rk1 = np.array([self.stimulus_filter[cellnum][7,7,:] for cellnum in range(self.n_cells)])


    def plot_mosaic(self, ax, n_sd=1.5, marker_size=0.3, marker_color='blue'):

        if n_sd > 0:
            for cellnum in range(self.n_cells):
                currcell = self.rgc_spatial_params.iloc[cellnum]

                rf_ellipse = Ellipse(xy=(currcell.center_x, currcell.center_y),
                                     width=currcell.sd_x * 2 * n_sd,
                                     height=currcell.sd_y * 2 * n_sd,
                                     angle=np.rad2deg(currcell.center_rot))

                rf_ellipse.set_edgecolor('black')
                if cellnum in ParasolMosaic.bad_indices:
                    rf_ellipse.set_facecolor('grey')
                else:
                    rf_ellipse.set_facecolor('none')
                ax.add_artist(rf_ellipse)

                plt.text(x=currcell.center_x, y=currcell.center_y, s=str(cellnum))

        ax.scatter(self.rgc_spatial_params.center_x, self.rgc_spatial_params.center_y, s=marker_size, c=marker_color)

    def plot_filters(self, cellnum, spatial_filter_t=1):

        postspike = self.postspike_filter[cellnum]
        stimfilt = self.stimulus_filter[cellnum]

        plt.subplots(1, 2, figsize=(16,8))

        plt.subplot(121)
        sns.heatmap(stimfilt[:,:,spatial_filter_t], square=True)

        plt.subplot(122)
        plt.title('Postspike filter (exponentiated)')
        plt.plot(range(len(postspike)), np.exp(postspike))

        plt.suptitle('Cell number %d' % cellnum)
        plt.show()

    def reshape_stimulus_filter(self):
        pass

    def interpolate_postspike_filter(self, video_fps, cellnum, ip_kind='cubic'):

        postspike_sec = ParasolMosaic.tsteps_postspike_filter / ParasolMosaic.sampling_rate
        postspike_video = postspike_sec * video_fps
        interpol_x = np.linspace(0, postspike_video, ParasolMosaic.tsteps_postspike_filter)
        ip_postspike_filter = interp1d(interpol_x, self.postspike_filter[cellnum][:,0], kind=ip_kind)

        return ip_postspike_filter

    def interpolate_temporal_filter(self, video_fps, cellnum, ip_kind='linear'):

        stimulusfilter_sec = ParasolMosaic.tsteps_stimulusfilter / ParasolMosaic.sampling_rate
        stimulusfilter_video = stimulusfilter_sec * video_fps
        interpol_x = np.linspace(0, stimulusfilter_video, ParasolMosaic.tsteps_stimulusfilter)
        #ip_stimulusfilter = interp1d(interpol_x, self.stimulus_filter[cellnum][7,7,:], kind=ip_kind)
        ip_stimulusfilter = interp1d(interpol_x, self.time_rk1[cellnum][:,0], kind=ip_kind)

        return ip_stimulusfilter, stimulusfilter_video//1

    def DoG2D_fixed_surround(self, xy_tuple, amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes,
                             sur_ratio, offset):
        '''
        DoG model with xo, yo, theta for surround coming from center.
        '''
        (x_fit, y_fit) = xy_tuple
        acen = (np.cos(orientation_center) ** 2) / (2 * semi_xc ** 2) + (np.sin(orientation_center) ** 2) / (
                    2 * semi_yc ** 2)
        bcen = -(np.sin(2 * orientation_center)) / (4 * semi_xc ** 2) + (np.sin(2 * orientation_center)) / (
                    4 * semi_yc ** 2)
        ccen = (np.sin(orientation_center) ** 2) / (2 * semi_xc ** 2) + (np.cos(orientation_center) ** 2) / (
                    2 * semi_yc ** 2)

        asur = (np.cos(orientation_center) ** 2) / (2 * sur_ratio * semi_xc ** 2) + (
                    np.sin(orientation_center) ** 2) / (2 * sur_ratio * semi_yc ** 2)
        bsur = -(np.sin(2 * orientation_center)) / (4 * sur_ratio * semi_xc ** 2) + (np.sin(2 * orientation_center)) / (
                    4 * sur_ratio * semi_yc ** 2)
        csur = (np.sin(orientation_center) ** 2) / (2 * sur_ratio * semi_xc ** 2) + (
                    np.cos(orientation_center) ** 2) / (2 * sur_ratio * semi_yc ** 2)

        ## Difference of gaussians
        model_fit = offset + \
                    amplitudec * np.exp(
            - (acen * ((x_fit - xoc) ** 2) + 2 * bcen * (x_fit - xoc) * (y_fit - yoc) + ccen * ((y_fit - yoc) ** 2))) - \
                    amplitudes * np.exp(
            - (asur * ((x_fit - xoc) ** 2) + 2 * bsur * (x_fit - xoc) * (y_fit - yoc) + csur * ((y_fit - yoc) ** 2)))

        return model_fit.ravel()

    def fit_rgc(self, cellnum):

        rgc_params = self.rgc_spatial_params.iloc[cellnum]
        rgc_spatial_filter = self.space_rk1[cellnum]
        #rgc_spatial_filter = self.stimulus_filter[cellnum][:, :, 2]

        p0 = np.array([1, 7, 7, 3, 3, rgc_params['center_rot'], 0.1, 3, 0])
        boundaries = (np.array([0, -np.inf, -np.inf, 0, 0, -2 * np.pi, 0, 1, -np.inf]),
                      np.array([1, np.inf, np.inf, np.inf, np.inf, 2 * np.pi, 1, np.inf, np.inf]))
        x_position_indices = np.arange(0, 13, 1)
        y_position_indices = np.arange(0, 13, 1)
        x_grid, y_grid = np.meshgrid(x_position_indices, y_position_indices)

        popt, pcov = opt.curve_fit(self.DoG2D_fixed_surround, (x_grid, y_grid), rgc_spatial_filter.ravel(), p0=p0,
                                   bounds=boundaries)

        return popt

    def fitted_vs_original(self, cellnum, n_pos=13, fps=120):  # TODO - Time rescaling

        # First fit the spatiotemporal filter
        # original_filter = self.stimulus_filter[cellnum][:, :, 2]
        original_filter = self.space_rk1[cellnum]
        fitted_params = self.fit_rgc(cellnum)

        x_position_indices = np.linspace(0, 13, n_pos)
        y_position_indices = np.linspace(0, 13, n_pos)
        x_grid, y_grid = np.meshgrid(x_position_indices, y_position_indices)
        fitted_func = self.DoG2D_fixed_surround((x_grid, y_grid), *fitted_params)

        # Second get a temporal filter
        temporal_filter, n_steps_temporal = self.interpolate_temporal_filter(fps, cellnum)
        print(n_steps_temporal)

        # Compare spatiotemporal filter energy
        fitted_func_1d = np.reshape(fitted_func, (n_pos*n_pos, 1))
        #fitted_spatiotemp = fitted_func_1d * self.time_rk1[cellnum].T
        temporal_filter_discrete = np.array([temporal_filter(np.arange(0,n_steps_temporal,1))])
        fitted_spatiotemp = fitted_func_1d * temporal_filter_discrete

        print('Original filter energy:')
        filter_integral = np.sum(self.stimulus_filter[cellnum].ravel()) / (13*13*15)
        print(filter_integral)

        print('Fitted spatiotemp energy:')
        fitted_filter_integral = np.sum(fitted_spatiotemp) / (n_pos*n_pos*n_steps_temporal)
        print(fitted_filter_integral)

        fitted_spatiotemp = fitted_spatiotemp * (filter_integral/fitted_filter_integral)

        print('Corrected fitted spatiotemp energy:')
        print(np.sum(fitted_spatiotemp) / (n_pos*n_pos*n_steps_temporal))


    def create_spatiotemporal_filter(self, cellnum, n_pos=13, savefig=False):  # fps fixed at 120
        # First fit the spatiotemporal filter
        # original_filter = self.stimulus_filter[cellnum][:, :, 2]
        original_filter = self.space_rk1[cellnum]
        fitted_params = self.fit_rgc(cellnum)


        x_position_indices = np.linspace(0, 13, n_pos)
        y_position_indices = np.linspace(0, 13, n_pos)

        x_grid, y_grid = np.meshgrid(x_position_indices, y_position_indices)
        fitted_func = self.DoG2D_fixed_surround((x_grid, y_grid), *fitted_params)

        # Second get a temporal filter
        #temporal_filter, n_steps_temporal = self.interpolate_temporal_filter(fps, cellnum)
        #print(n_steps_temporal)

        # Compare spatiotemporal filter energy
        fitted_func_1d = np.reshape(fitted_func, (n_pos * n_pos, 1))
        fitted_spatiotemp = fitted_func_1d * self.time_rk1[cellnum].T
        #temporal_filter_discrete = np.array([temporal_filter(np.arange(0, n_steps_temporal, 1))])
        #fitted_spatiotemp = fitted_func_1d * temporal_filter_discrete

        if savefig is False:
            return fitted_params

        else:
            print('Original filter energy:')
            filter_integral = np.sum(self.stimulus_filter[cellnum].ravel()) / (13 * 13)
            print(filter_integral)

            print('Fitted spatiotemp energy:')
            fitted_filter_integral = np.sum(fitted_spatiotemp) / (n_pos * n_pos)
            print(fitted_filter_integral)

            fitted_spatiotemp = fitted_spatiotemp * (filter_integral / fitted_filter_integral)

            print('Corrected fitted spatiotemp energy:')
            print(np.sum(fitted_spatiotemp) / (n_pos * n_pos))


            # Plot spatiotemporal filters
            vmax = np.max(np.concatenate((self.stimulus_filter[cellnum].ravel(), fitted_spatiotemp.ravel()), axis=0))
            vmin = np.min(np.concatenate((self.stimulus_filter[cellnum].ravel(), fitted_spatiotemp.ravel()), axis=0))

            fig, ax = plt.subplots(1,2)
            spatiotemp_filter = np.reshape(self.stimulus_filter[cellnum], (13*13, 15))
            sns.heatmap(spatiotemp_filter, ax=ax[0], vmax=vmax, vmin=vmin)

            #fitted_func = np.reshape(fitted_func, (13,13))
            sns.heatmap(fitted_spatiotemp, ax=ax[1], vmax=vmax, vmin=vmin)

            plt.savefig('./filterfigs/onparasol_'+str(cellnum)+'.png')


    # def create_spatial_filter(self, cellnum):
    #     rgc_params = self.rgc_spatial_params[cellnum]
    #     z_center = rgc_params['z_center']
    #     sigma_center = rgc_params['sigma_center']
    #     sigma_surround = rgc_params['sigma_surround']
    #     factor_surround = rgc_params['factor_surround']
    #
    #     filter_height = video_height
    #     filter_width = video_width
    #
    #     x, y = np.meshgrid(np.linspace(-video_width_deg / 2, video_width_deg / 2, filter_width),
    #                        -np.linspace(-video_height_deg / 2, video_height_deg / 2, filter_height))  # Why minus?
    #
    #     d = np.sqrt((x - z_center.real) ** 2 + (y - z_center.imag) ** 2)
    #     filter_center = np.exp(-(d ** 2 / (2.0 * sigma_center ** 2)))
    #     filter_surround = factor_surround * np.exp(-(d ** 2 / (2.0 * sigma_surround ** 2)))
    #
    #     dog_filter = filter_center - filter_surround
    #
    #     return dog_filter

    def feed_stimulus(self, stim_video, cellnum, nonlinearity=np.exp):

        # Get spatial filter parameters
        fitted_spatial_params = self.create_spatiotemporal_filter(cellnum)
        # amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes,sur_ratio, offset

        # Convert spatial kernel to visual space
        c = 1 / ParasolMosaic.data_pix_per_deg
        fitted_spatial_params_vs = fitted_spatial_params
        fitted_spatial_params_vs[3] = fitted_spatial_params_vs[3] * c
        fitted_spatial_params_vs[4] = fitted_spatial_params_vs[4] * c

        fitted_spatial_params_vs[1] = self.rgc_spatial_params.iloc[cellnum]['center_x']
        fitted_spatial_params_vs[2] = self.rgc_spatial_params.iloc[cellnum]['center_y']

        # Prepare spatial kernel
        filter_height = stim_video.video_height
        filter_width = stim_video.video_width

        x_grid, y_grid = np.meshgrid(np.linspace(stim_video.video_xmin_deg, stim_video.video_xmax_deg, filter_width),
                                     np.linspace(stim_video.video_ymin_deg, stim_video.video_ymax_deg, filter_height))

        spatial_kernel = np.array([self.DoG2D_fixed_surround((x_grid, y_grid), *fitted_spatial_params)]).T

        # Make the spatiotemporal kernel
        # TODO - time rescaling depending on video fps
        spatiotemporal_kernel = spatial_kernel * self.time_rk1[cellnum].T

        # Scale the values so that sum(original stimulus filter) = sum(fitted/rescaled stimulus filter)
        original_filter_integral = np.sum(self.stimulus_filter[cellnum].ravel())
        fitted_filter_integral = np.sum(spatiotemporal_kernel.ravel())
        spatiotemporal_kernel = (original_filter_integral/fitted_filter_integral) * spatiotemporal_kernel

        # Deal with video
        stim_video_2d = stim_video.get_2d_video()
        filtered_stim = fftconv(stim_video_2d, spatiotemporal_kernel, mode='valid')

        return nonlinearity(filtered_stim.ravel()/255) + self.tonicdrive[cellnum]


class SimpleVideo(object):

    def __init__(self, video_path_and_pattern, fps=25, pix_per_deg=4.5, video_center_pc=0+0j):  # defaults from catcam dataset

        self.video = skimage.io.ImageCollection(video_path_and_pattern)
        self.fps = fps
        self.pix_per_deg = pix_per_deg
        self.deg_to_px = lambda x: (x * self.pix_per_deg) // 1

        self.video_n_frames = len(self.video)
        self.video_width = self.video[0].shape[1]
        self.video_height = self.video[1].shape[0]
        self.video_width_deg = self.video_width / self.pix_per_deg
        self.video_height_deg = self.video_height / self.pix_per_deg

        self.video_xmin_deg = video_center_pc.real - self.video_width_deg /2
        self.video_xmax_deg = video_center_pc.real + self.video_width_deg /2
        self.video_ymin_deg = video_center_pc.imag - self.video_height_deg /2
        self.video_ymax_deg = video_center_pc.imag + self.video_height_deg /2
        #video_center = self.video_width // 2 + self.video_height // 2 * 1j
        #video_center_pc = video_center / self.pix_per_deg

    def show_frame(self, frame_ix=0):
        plt.imshow(self.video[frame_ix])
        plt.show()

    def get_2d_video(self):
        stim_video_2d = np.reshape(self.video, (self.video_n_frames,
                                                self.video_height*self.video_width)).T  # pixels as rows, time as cols
        return stim_video_2d



if __name__ == '__main__':

    mosaic = ParasolMosaic()
    a = SimpleVideo('./movie01/Catt0*.tif', video_center_pc=35+15j)
    convcat = mosaic.feed_stimulus(a, 5)
    plt.plot(range(len(convcat)), convcat)
    plt.show()


    # video_grating = vs.ConstructStimulus(video_center_pc=35+15j, pattern='sine_grating', temporal_frequency=8, duration_seconds=2, fps=120,
    #                                      orientation=45, image_width=320, image_height=240, pix_per_deg=4.5)
    #
    # convgrating = mosaic.feed_stimulus(video_grating, 5)
    # plt.plot(range(len(convgrating)), convgrating)
    #
    # plt.show()

    # for n in range(mosaic.n_cells):
    #    mosaic.create_spatiotemporal_filter(n, 26)
    #mosaic.create_spatiotemporal_filter(5, 26, savefig=False)

    # fig,ax = plt.subplots(1,1)
    # mosaic.plot_mosaic(ax=ax)
    # plt.show()

    #
    #a.show_frame()
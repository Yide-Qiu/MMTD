from collections import defaultdict
import itertools
import json
import pdb
import random

import torch
import numpy as np

from .data import SceneRow, TrackRow_ship_obs, TrackRow_ship_gt


class Reader_ship_xysc(object):

    def __init__(self, input_file=None, obs=None, scene_type=None, image_file=None):
        if scene_type is not None and scene_type not in {'rows', 'paths', 'tags'}:
            raise Exception('scene_type not supported')
        self.scene_type = scene_type

        self.tracks_by_frame = defaultdict(list)
        self.gt_tracks_by_frame = defaultdict(list)
        self.scenes_by_id = dict()
        if input_file is not None:
            self.read_file(input_file)
        elif obs is not None:
            self.read_obs(obs)
        else:
            raise('please define a data load function')
        self.vis_len = 30
    # for mot  version 1
    def read_obs(self, obs):
        scene_id = 0
        # obs --> array([1, 11.593051544424076, -134.30233669878012, 11.59359053112907,
        #    -134.30600477357106, 2.7215236128359788, 50.18319849804168, 'elec',
        #    'satellite_3089271696']
        for i in range(obs.shape[0]):
            f_start = np.min(obs[i, :, 0])
            f_end = np.max(obs[i, :, 0])
            for j in range(obs.shape[1]):
                f, y_b, x_r, y_t, x_l ,s, c, source, satellite_id = obs[i, j]
                row = TrackRow_ship_obs(f, -1, i, x_l, y_t, s, c)
                self.tracks_by_frame[row.frame].append(row)
            row = SceneRow(scene_id, -1, i, f_start, f_end, None, None)
            self.scenes_by_id[row.scene] = row
            scene_id += 1
        # pdb.set_trace()

    # # for mot  version 2
    # def read_obs(self, obs):
    #     scene_id = 0
    #     # obs --> array([tlwhsc])
    #     for i in range(obs.shape[0]):
    #         f_start = np.min(obs[i, :, 0])
    #         f_end = np.max(obs[i, :, 0])
    #         for j in range(obs.shape[1]):
    #             t, l, w, h ,s, c, f = obs[i, j]
    #             row = TrackRow_ship_obs(f, -1, i, t, l, s, c)
    #             self.tracks_by_frame[row.frame].append(row)
    #         row = SceneRow(scene_id, -1, i, f_start, f_end, None, None)
    #         self.scenes_by_id[row.scene] = row
    #         scene_id += 1
    #     # pdb.set_trace()

    def read_file(self, input_file):
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line)
                # 观测轨迹点数据
                track = line.get('track')
                # "track": {"f": 0, "o": 0, "p": "137711000", "ship_type": "100000019", "length": 194.0, "width": 33.0,
                # "lat_bottle": 21.305324723274204, "lon_right": -165.43946643912543, "lat_top": 21.305621165961952, "lon_left": -165.44133700179958,
                # "sog": 2.0446678053171508, "cog": 46.48277310074681, "source": "elec"}}
                if track is not None:
                    row = TrackRow_ship_obs(track['f'], track['o'], track['p'], track['lon_left'], track['lat_top'],
                                    track['sog'], track['cog'])
                    self.tracks_by_frame[row.frame].append(row)  # size = 100 * N
                    continue

                scene = line.get('scene')
                if scene is not None:
                    row = SceneRow(scene['id'], scene['o'], scene['p'], scene['s'], scene['e'], \
                                   scene.get('fps'), scene.get('tag'))
                    self.scenes_by_id[row.scene] = row  # 这个没问题，是按顺序的  len=390

                # todo  GT轨迹点数据
                track = line.get('gt_track')
                # {"gt_track": {"f": 0, "o": 0, "p": "137711000", "ship_type": "100000019",
                # "lat_top": 21.29125578588189, "lon_left": -165.48647113934788, "sog": 0.0, "cog": 44.980908424081896}}
                if track is not None:
                    row = TrackRow_ship_gt(track['f'], track['o'], track['p'], track['lon_left'], track['lat_top'],
                                    track['sog'], track['cog'])
                    self.gt_tracks_by_frame[row.frame].append(row)  # size = 100 * N
                    continue
        # import pdb
        # pdb.set_trace()
    def scenes(self, randomize=False, limit=0, ids=None, sample=None):
        scene_ids = self.scenes_by_id.keys()
        if ids is not None:
            scene_ids = ids
        if randomize:
            scene_ids = list(scene_ids)
            random.shuffle(scene_ids)
        if limit:
            scene_ids = itertools.islice(scene_ids, limit)
        if sample is not None:
            scene_ids = random.sample(scene_ids, int(len(scene_ids) * sample))
            # import pdb
            # pdb.set_trace()
        for scene_id in scene_ids:
            yield self.scene(scene_id)
    @staticmethod
    def track_rows_to_paths(primary_mmsi, track_rows):
        primary_path = []
        other_paths = defaultdict(list)
        for row in track_rows:
            if str(row.mmsi) == str(primary_mmsi):
                primary_path.append(row)
                continue
            other_paths[row.mmsi].append(row)
        if primary_path == []:
            pdb.set_trace()

        return [primary_path] + list(other_paths.values())
    @staticmethod
    def paths_to_xy(paths):
        """Convert paths to numpy array with nan as blanks."""
        frames = set(r.frame for r in paths[0])
        mmsi_list = set(row.mmsi
                          for path in paths
                          for row in path if row.frame in frames)
        # if len(frames) != 30:
        #     pdb.set_trace()
        # print(mmsi_list)
        # pdb.set_trace()
        paths = [path for path in paths if path[0].mmsi in mmsi_list]
        frames = sorted(frames)
        mmsi_list = list(mmsi_list)

        frame_to_index = {frame: i for i, frame in enumerate(frames)}
        input_token_num = 3

        xy = np.full((len(frames), len(mmsi_list), 4 * input_token_num), 0.0, dtype=np.float32)
        # pdb.set_trace()

        for ped_index, path in enumerate(paths):
            for row in path:
                if row.frame not in frame_to_index:
                    # pdb.set_trace()
                    continue
                # pdb.set_trace()
                entry = xy[frame_to_index[row.frame]][ped_index]

                ## extract trajectory
                entry[0] = row[3]
                entry[1] = row[4]
                entry[2] = 0.0
                entry[3] = 0.0
                entry[4] = row[5]
                entry[5] = 0.0
                entry[6] = 0.0
                entry[7] = 0.0
                entry[8] = row[6]
                entry[9] = 0.0
                entry[10] = 0.0
                entry[11] = 0.0
        # pdb.set_trace()
        contains_nan = torch.isnan(torch.tensor(xy)).any()
        if contains_nan:
            # print(mmsi, scene_id, filename)
            pdb.set_trace()
        return xy, mmsi_list[0], frames

    def scene(self, scene_id, total_joints_dim=66):
        scene = self.scenes_by_id.get(scene_id)
        if scene is None:
            raise Exception('scene with that id not found')
        # if scene.end - scene.start +1 != 30:
        #     pdb.set_trace()
        frames_obs = range(scene.start, scene.start + self.vis_len)
        frames_pred = range(scene.start + self.vis_len, scene.end + 1)
        track_rows_vis = [r for frame in frames_obs for r in self.tracks_by_frame.get(frame, [])]
        track_rows_pred = [r for frame in frames_pred for r in self.gt_tracks_by_frame.get(frame, [])]
        track_rows = track_rows_vis + track_rows_pred
        # pdb.set_trace()
        filtered_rows = []
        for row in track_rows:
            if int(row.obs) == int(scene.obs):
                filtered_rows.append(row)
        track_rows = filtered_rows
        # pdb.set_trace()
        # return as rows
        if self.scene_type == 'rows':
            return scene_id, scene.mmsi, track_rows

        # return as paths
        paths = self.track_rows_to_paths(scene.mmsi, track_rows)
        if self.scene_type == 'paths':
            return scene_id, paths

        ## return with scene tag

        if self.scene_type == 'tags':
            return scene_id, scene.tag, self.paths_to_xy(paths)

        # return a numpy array
        return scene_id, self.paths_to_xy(paths)




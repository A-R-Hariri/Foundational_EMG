import json
import os
import h5py
import numpy as np
import warnings

from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler


# ======== CONSTANTS ========

GESTURE_MAP = {
    "noGesture": 0,
    "fist": 1,
    "waveIn": 2,
    "waveOut": 3,
    "open": 4,
    "pinch": 5,
}


# ======== JSON → H5 ========

def _sorted_keys(d):
    return sorted(d.keys(), key=lambda k: int(k.split("_")[1]))


def _user_id_from_dir(user_dir, offset):
    return int(user_dir.replace("user", "")) - 1 + offset

def _get_pb_pe(entry):
    if "groundTruth" not in entry:
        return -1, -1
    gt = np.asarray(entry["groundTruth"], dtype=np.int8)
    idx = np.where(gt != 0)[0]
    if idx.size == 0:
        return -1, -1
    pb = idx[0]
    pe = idx[-1] + 1
    return pb, pe

def process_user_json(
    json_path,
    out_path,
    subject_id,
    use_testing_samples,
):
    with open(json_path, "r") as f:
        data = json.load(f)

    blocks = [data["trainingSamples"]]
    if use_testing_samples:
        blocks.append(data["testingSamples"])

    rep_counter = {g: 0 for g in GESTURE_MAP.values()}

    with h5py.File(out_path, "w") as h5:
        reps_grp = h5.create_group("reps")

        for block in blocks:
            for k in _sorted_keys(block):
                entry = block[k]

                if "emg" not in entry:
                    warnings.warn(
                        f"Missing EMG: subject={subject_id}"
                    )
                    continue

                pb, pe = _get_pb_pe(entry)

                gesture = GESTURE_MAP[entry["gestureName"]]
                rep_id = rep_counter[gesture]
                rep_counter[gesture] += 1

                emg = np.stack(
                    [entry["emg"][f"ch{i}"] for i in range(1, 9)],
                    axis=1,
                ).astype(np.float32)

                rep_grp = reps_grp.create_group(
                    f"rep_g{gesture}_r{rep_id:02d}"
                )
                rep_grp.create_dataset("emg", data=emg)
                rep_grp.create_dataset("gesture", data=gesture)
                rep_grp.create_dataset("subject", data=subject_id)
                rep_grp.create_dataset("rep", data=rep_id)
                rep_grp.create_dataset("pb", data=pb)
                rep_grp.create_dataset("pe", data=pe)

    print(
        f"Finished subject={subject_id} | "
        f"reps_per_gesture={rep_counter} | "
        f"out={out_path}"
    )


# ======== DATASET WALKER ========

def process_dataset_epn612(root_in, root_out):
    splits = {
        "trainingJSON": (True, 0),
        "testingJSON": (False, 306),
    }

    for split, (use_testing_samples, offset) in splits.items():
        in_split = os.path.join(root_in, split)
        out_split = os.path.join(
            root_out, split.replace("JSON", "")
        )
        os.makedirs(out_split, exist_ok=True)

        user_dirs = sorted(
            d for d in os.listdir(in_split)
            if os.path.isdir(os.path.join(in_split, d))
        )

        print(f"\n=== Processing {split} ===")

        for user_dir in user_dirs:
            subject_id = _user_id_from_dir(user_dir, offset)

            json_path = os.path.join(
                in_split, user_dir, f"{user_dir}.json"
            )
            out_path = os.path.join(
                out_split, f"{user_dir}.h5"
            )

            process_user_json(
                json_path=json_path,
                out_path=out_path,
                subject_id=subject_id,
                use_testing_samples=use_testing_samples,
            )


# ======== LIBEMG DATASET ========

class EMGEPN612(Dataset):
    def __init__(self, dataset_folder="EPN612"):
        super().__init__(
            sampling={"myo": 200},
            num_channels={"myo": 8},
            recording_device=["myo"],
            num_subjects=612,
            gestures=GESTURE_MAP,
            num_reps="training: 50 reps/gesture | testing: 25 reps/gesture",
            description="EPN-612 EMG dataset (JSON)",
            citation="",
        )
        self.dataset_folder = dataset_folder


    def _get_odh(
        self,
        processed_root,
        subjects,
        segment,
        relabel_seg,
        channel_last,
    ):
        odhs = []

        for split in ["training", "testing"]:
            split_dir = os.path.join(processed_root, split)
            odh = OfflineDataHandler()

            odh.subjects = []
            odh.classes = []
            odh.reps = []
            odh.base_class = []
            odh.extra_attributes = [
                "subjects", "classes", 
                "reps", "base_class"]

            for file in sorted(os.listdir(split_dir)):
                with h5py.File(
                    os.path.join(split_dir, file), "r") as f:

                    for rep_name in f["reps"]:
                        r = f["reps"][rep_name]

                        subject = int(r["subject"][()])
                        if subjects is not None and subject not in subjects:
                            continue

                        gst = int(r["gesture"][()])
                        rep_id = int(r["rep"][()])
                        emg = r["emg"][:]
                        emg = emg.astype(np.float32)

                        if segment and gst != 0:
                            pb = int(r["pb"][()])
                            pe = int(r["pe"][()])
                            if pb < 0 or pe < 0:
                                pb, pe = None, None
                            emg = emg[pb:pe]

                        if not len(emg):
                            continue

                        if not channel_last:
                            emg = emg.T

                        odh.data.append(emg)
                        odh.classes.append(np.ones((len(emg), 1), 
                                                   dtype=np.int64) * gst)
                        odh.subjects.append(np.ones((len(emg), 1), 
                                                    dtype=np.int64) * subject)
                        odh.reps.append(np.ones((len(emg), 1), 
                                                dtype=np.int64) * rep_id)
                        odh.base_class.append(np.ones((len(emg), 1), 
                                                      dtype=np.int64) * gst)

                        if segment and gst != 0 and relabel_seg is not None \
                            and pb is not None and pe is not None:
                            relabel_seg = int(relabel_seg) 
        
                            emg2 = r["emg"][:pb]
                            if not len(emg2):
                                continue
                            if not channel_last:
                                emg2 = emg2.T
                            odh.data.append(emg2)
                            odh.classes.append(np.ones((len(emg2), 1), 
                                                       dtype=np.int64) * relabel_seg)
                            odh.subjects.append(np.ones((len(emg2), 1), 
                                                        dtype=np.int64) * subject)
                            odh.reps.append(np.ones((len(emg2), 1), 
                                                    dtype=np.int64) * rep_id)
                            odh.base_class.append(np.ones((len(emg2), 1), 
                                                       dtype=np.int64) * gst)
                            
                            emg2 = r["emg"][pe:]
                            if not len(emg2):
                                continue
                            if not channel_last:
                                emg2 = emg2.T
                            odh.data.append(emg2)
                            odh.classes.append(np.ones((len(emg2), 1), 
                                                       dtype=np.int64) * relabel_seg)
                            odh.subjects.append(np.ones((len(emg2), 1), 
                                                        dtype=np.int64) * subject)
                            odh.reps.append(np.ones((len(emg2), 1), 
                                                    dtype=np.int64) * rep_id)
                            odh.base_class.append(np.ones((len(emg2), 1), 
                                                       dtype=np.int64) * gst)

                            

            odhs.append(odh)

        return odhs


    def prepare_data(
        self,
        split=False,
        segment=True,
        relabel_seg=None,
        channel_last=True,
        subjects=None,
    ):
        processed = self.dataset_folder + "_PROCESSED"

        if not os.path.exists(processed):
            process_dataset_epn612(
                self.dataset_folder, processed
            )

        odh_tr, odh_te = self._get_odh(
            processed,
            subjects,
            segment,
            relabel_seg,
            channel_last,
        )

        return (
            {"All": odh_tr + odh_te, "Train": odh_tr, "Test": odh_te}
            if split else odh_tr + odh_te
        )

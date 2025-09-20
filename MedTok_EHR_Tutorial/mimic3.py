import os
from typing import Optional, List, Dict, Tuple, Union

import pandas as pd

from data import Event, Visit, Patient
from pyhealth.datasets import BaseEHRDataset
from pyhealth.datasets.utils import strptime

# TODO: add other tables


class MIMIC3Dataset(BaseEHRDataset):
    """Base dataset for MIMIC-III dataset.

    The MIMIC-III dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    The basic information is stored in the following tables:
        - PATIENTS: defines a patient in the database, subject_id.
        - ADMISSIONS: defines a patient's hospital admission, hadm_id.

    We further support the following tables:
        - DIAGNOSES_ICD: contains ICD-9 diagnoses (ICD9CM code) for patients.
        - PROCEDURES_ICD: contains ICD-9 procedures (ICD9PROC code) for patients.
        - PRESCRIPTIONS: contains medication related order entries (ndc code)
            for patients.
        - LABEVENTS: contains laboratory measurements (MIMIC3_itemid code)
            for patients

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                (1) a str of the target code vocabulary;
                (2) a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method.
            Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> dataset = MIMIC3Dataset(
        ...         root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...         tables=["DIAGNOSES_ICD", "PRESCRIPTIONS"],
        ...         code_mapping={"ndc": ("ATC", {"target_kwargs": {"level": 3}})},
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PATIENTS and ADMISSIONS tables.

        Will be called in `self.parse_tables()`

        Docs:
            - PATIENTS: https://mimic.mit.edu/docs/iii/tables/patients/
            - ADMISSIONS: https://mimic.mit.edu/docs/iii/tables/admissions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id which is updated with the mimic-3 table result.

        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            dtype={"subject_id": str},
            #nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"subject_id": str, "hadm_id": str},
        )
        # merge patient and admission tables
        df = pd.merge(patients_df, admissions_df, on="subject_id", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["subject_id", "admittime", "dischtime"], ascending=True)
        # group by patient
        df_group = df.groupby("subject_id")

        ##read ICUSTAYS TABLES
        icustays_df = pd.read_csv(
            os.path.join(self.root, "ICUSTAYS.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icustay_id": str},
        )
        df_admission_icustay=pd.merge(df, icustays_df, on=["subject_id", "hadm_id"], how="inner")
        df_admission_icustay = df_admission_icustay.sort_values(["subject_id", "admittime", "intime"], ascending=True)
        df_group = df_admission_icustay.groupby("subject_id")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            patient = Patient(
                patient_id=p_id,
                birth_datetime=strptime(p_info["dob"].values[0]),
                death_datetime=strptime(p_info["dod_hosp"].values[0]),
                gender=p_info["gender"].values[0],
                ethnicity=p_info["ethnicity"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("hadm_id"):
                #print(v_info['icustay_id'].values)
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info["admittime"].values[0]),
                    discharge_time=strptime(v_info["dischtime"].values[0]),
                    discharge_status=v_info["hospital_expire_flag"].values[0],
                    insurance=v_info["insurance"].values[0],
                    language=v_info["language"].values[0],
                    religion=v_info["religion"].values[0],
                    marital_status=v_info["marital_status"].values[0],
                    ethnicity=v_info["ethnicity"].values[0],
                    icustays_num = len(v_info['icustay_id'].values)
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.subject_id.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses DIAGNOSES_ICD table.

        Will be called in `self.parse_tables()`

        Docs:
            - DIAGNOSES_ICD: https://mimic.mit.edu/docs/iii/tables/diagnoses_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-III does not provide specific timestamps in DIAGNOSES_ICD
                table, so we set it to None.
        """
        table = "DIAGNOSES_ICD"
        self.code_vocs["conditions"] = "ICD9CM"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd9_code": str},
        )
        # drop records of the other patients
        df = df[df["subject_id"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd9_code"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")


        ##read ICUSTAYS
        '''icustays_df = pd.read_csv(
            os.path.join(self.root, "ICUSTAYS.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icustay_id": str},
        )
        df = pd.merge(df, icustays_df, on=["subject_id"], how="inner")
        diagnosis_icustays = '''

        # parallel unit of diagnosis (per patient)
        def diagnosis_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code in v_info["icd9_code"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9CM",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: diagnosis_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PROCEDURES_ICD table.

        Will be called in `self.parse_tables()`

        Docs:
            - PROCEDURES_ICD: https://mimic.mit.edu/docs/iii/tables/procedures_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-III does not provide specific timestamps in PROCEDURES_ICD
                table, so we set it to None.
        """
        table = "PROCEDURES_ICD"
        self.code_vocs["procedures"] = "ICD9PROC"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd9_code": str},
        )
        # drop records of the other patients
        df = df[df["subject_id"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "seq_num", "icd9_code"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of procedure (per patient)
        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code in v_info["icd9_code"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9PROC",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: procedure_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PRESCRIPTIONS table.

        Will be called in `self.parse_tables()`

        Docs:
            - PRESCRIPTIONS: https://mimic.mit.edu/docs/iii/tables/prescriptions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "PRESCRIPTIONS"
        self.code_vocs["drugs"] = "ndc"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"subject_id": str, "hadm_id": str, "ndc": str},
        )
        # drop records of the other patients
        df = df[df["subject_id"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "ndc"])
        # sort by start date and end date
        df = df.sort_values(
            ["subject_id", "hadm_id", "startdate", "enddate"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit for prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code in zip(v_info["startdate"], v_info["ndc"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ndc",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: prescription_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses LABEVENTS table.

        Will be called in `self.parse_tables()`

        Docs:
            - LABEVENTS: https://mimic.mit.edu/docs/iii/tables/labevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "LABEVENTS"
        self.code_vocs["labs"] = "MIMIC3_itemid"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "itemid": str},
        )
        # drop records of the other patients
        df = df[df["subject_id"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "itemid"])
        # sort by charttime
        df = df.sort_values(["subject_id", "hadm_id", "charttime"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit for lab (per patient)
        def lab_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code, value, unit, flag in zip(v_info["charttime"], v_info["itemid"], v_info["value"], v_info["valueUOM"], v_info["flag"]): 
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MIMIC3_itemid",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                        value=value,
                        unit=unit, 
                        flag=flag
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: lab_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients
    
    def parse_icustays(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses ICUSTAYS table.

        Will be called in `self.parse_tables()`

        Docs:
            - ICUSTAYS: https://mimic.mit.edu/docs/iii/tables/icustays/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "ICUSTAYS"
        self.code_vocs["icustays"] = 'icustay'
        #print("read_icustays")
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icustay_id": str},
        )
        #print("read_icustays")
        # drop records of the other patients
        df = df[df["subject_id"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icustay_id"])
        # sort by start date and end date
        df = df.sort_values(
            ["subject_id", "hadm_id", "icustay_id", "intime", "outtime"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("subject_id")

        def icustays_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, icustay_id in zip(v_info["intime"], v_info["icustay_id"]):
                    event = Event(
                        code = None,
                        icustay_id=icustay_id,
                        table=table,
                        visit_id=v_id,
                        patient_id=p_id,
                        vocabulary='icustay',
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: icustays_unit(x.subject_id.unique()[0], x)
        )
        patients = self._add_events_to_patient_dict(patients, group_df)



if __name__ == "__main__":
    dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/mimiciii-demo/1.4/",
        tables=[
            "DIAGNOSES_ICD",
            "PROCEDURES_ICD",
            "PRESCRIPTIONS",
            "LABEVENTS",
        ],
        code_mapping={"ndc": "ATC"},
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()

    # dataset = MIMIC3Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    #     tables=["DIAGNOSES_ICD", "PRESCRIPTIONS"],
    #     dev=True,
    #     code_mapping={"ndc": ("ATC", {"target_kwargs": {"level": 3}})},
    #     refresh_cache=False,
    # )
    # print(dataset.stat())
    # print(dataset.available_tables)
    # print(list(dataset.patients.values())[4])

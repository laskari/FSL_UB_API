import torch, os

VERSION = "ub_v1.1"
ROOT = os.getcwd()
# ROOT = r"D:\project\FSL\new_codebase\FSL_UB_API"
artifact = 'artifacts'
LOG_DIR = "logs"
LOG_FILE = "UB_logs.log"
LOGFILE_DIR = os.path.join(ROOT, LOG_DIR, LOG_FILE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORY_MAPPING_PATH = os.path.join(ROOT, artifact, "notes.json")
MODEL_PATH = os.path.join(ROOT, artifact, "ub__100.pth")
UB_FORM_KEY_MAPPING = os.path.join(ROOT, artifact, "FSL_Forms_Keys.xlsx")
UB_AVERAGE_COORDINATE_PATH = os.path.join(ROOT, artifact, "average_coordinates_ub.xlsx")

KEYS_FROM_OLD = ['38_InsAddr1',
  '38_InsCity',
  '38_InsPostCode',
  '38_InsState',]

BBOX_DONUT_Mapping_Dict = {'1_Bill_Prov_Details': ['1_BillProvAddr1',
  '1_BillProvCity',
  '1_BillProvOrgName',
  '1_BillProvPhoneNumber',
  '1_BillProvPostCode',
  '1_BillProvState',
  '1_MediBillProvAddr1',
  '1_MediBillProvAddr2',
  '1_MediBillProvCity',
  '1_MediBillProvCredential',
  '1_MediBillProvFullName',
  '1_MediBillProvFullPost',
  '1_MediBillProvLName',
  '1_MediBillProvMI',
  '1_MediBillProvOrgName',
  '1_MediBillProvPhoneNumber',
  '1_MediBillProvPostCode',
  '1_MediBillProvPostCodeExt',
  '1_MediBillProvPrefix',
  '1_MediBillProvState',
  '1_MediBillProvSuffix',
  '1_MediBillProvvFName'],
 '2_PO_BOX': ['2_PayToProvAddr1',
  '2_PayToProvAddr2',
  '2_PayToProvCity',
  '2_PayToProvFedIdCode',
  '2_PayToProvFullPost',
  '2_PayToProvNPI',
  '2_PayToProvOrgName',
  '2_PayToProvPhoneNumber',
  '2_PayToProvPostCode',
  '2_PayToProvPostCodeExt',
  '2_PayToProvState'],
 '3_PAT CNTL AND MED REC': ['3_PatControlNumber', '3_PatMedicalRecordNo', 'MedicaidAmount'],
 '4_TYPE OF BILL': '4_TypeofBill',
 '5_FED_TAX_NO': ['5_BillProvFedIdCode', '5_MedicaidTaxId'],
 '6_STATEMENT_COVERS_PERIOD': ['6_StateCovFrom', '6_StateCovTo'],
 '7_Label': '',
 '8_PATIENT_NAME': ['8_PatFullName','8_PatIdentifier'],
 '9_PATIENT_ADDRESS': ['9_PatAddr1',
  '9_PatCity',
  '9_PatPostCode',
  '9_PatState'],
 '10_PATIENT_DOB': '10_PatDOB',
 '11_PATIENT_SEX': '11_PatSex',
 '12_ADMISSION_DETAILS': ['12_PatAdmissionDate',
  '13_PatAdmissionHour',
  '4_AdmissionType',
  '15_PatAdmissionSource',
  '14_PatAdmissionType'],
 '16_DISCHARGE_DETAILS': ['16_PatDischargeHour', '17_PatDischargeStatus'],
 '18_28_CONDITION_CODES': ['18_ConditionCode',
  '19_ConditionCode',
  '20_ConditionCode',
  '21_ConditionCode',
  '22_ConditionCode',
  '23_ConditionCode',
  '24_ConditionCode',
  '25_ConditionCode',
  '26_ConditionCode',
  '27_ConditionCode',
  '28_ConditionCode'],
 '29_State': '29_AccdtSt',
 '30_Label': '',
 '31_OCCURENCE_CODE_DATE': ['31_OccCode', '31_OccDate'],
 '32_OCCURENCE_CODE_DATE': ['32_OccCode', '32_OccDate'],
 '33_OCCURENCE_CODE_DATE': ['33_OccCode', '33_OccDate'],
 '34_OCCURENCE_CODE_DATE': ['34_OccCode', '34_OccDate'],
 '35_OCCURENCE_CODE_SPAN': ['35_OccFromDate',
  '35_OccSpanCode',
  '35_OccThruDate',
  'OccThruDate', 
  'OccSpanCode', 'OccFromDate'],
 '36_OCCURENCE_CODE_SPAN': ['36_OccFromDate',
  '36_OccSpanCode',
  '36_OccThruDate'],
 '37_Label': ' ',
 '38_INSURRENCE_DETAILS': ['38_InsAddr1',
  '38_InsCity',
  '38_InsFName',
  '38_InsFullName',
  '38_InsFullPost',
  '38_InsLName',
  '38_InsMI',
  '38_InsOrgName',
  '38_InsPostCode',
  '38_InsPrefix',
  '38_InsState',
  '38_InsSuffix'],
 '39_VALUE_CODES': ['39_ValueAmount', '39_ValueCode'],
 '40_VALUE_CODES': ['40_ValueAmount', '40_ValueCode'],
 '41_VALUE_CODES': ['41_ValueAmount', '41_ValueCode'],
 '42_49_TABLE': ['42_MedicaidPaidAmount',
  '42_RevCD',
  '43_Description',
  '43_NDC',
  '43_NDCUnits',
  '43_NDCUnitsQual',
  '44_HCPCS',
  '44_Modifier',
  '45_ServDate',
  '46_Unit',
  '47_Charges',
  '48_NonCovChrg'],
 '50_PAYER_NAME': ['50_Payer', '50_PayerIndicator'],
 '51_Health_Plan_No': '51_ProvId',
 '52_Label': '52_RelInfo',
 '53_ASG_BEN': '53_AsgBen',
 '54_PRIOR_PAYMENTS': '54_PriorPay',
 '55_Est_Amount_Due': '55_EstAmount',
 '56_NPI': ['56_BillProvNPI', '56_MediBillProvNPI'],
 '57_OTHER_PRV_ID': ['57_BillProvOtherIdQual', '57_BillProvOtherId'],
 '58_INSURED_NAME': '58_InsFullName',
 '59_P.REL': '59_InsPatRel',
 '60_INSURED_UNIQUE_ID': '60_InsIdCode',
 '61_GROUP_NAME': '61_GrpName',
 '62_INSURANCE_GROUP_NO': '62_GrpNum',
 '63_TREATMENT_AUTHORIZATION_CODES': '63_TreatAutoCode',
 '64_Document_Control_Number': '64_DocCtrlNum',
 '65_EMPLOYER_NAME': '65_EmpName',
 '67_DX': ['67_PrinDiagCode',
  '67_PrinPOA',
  '67A_OtherDiagCode',
  '67A_OtherDiagPOA',
  '67B_OtherDiagCode',
  '67B_OtherDiagPOA',
  '67C_OtherDiagCode',
  '67C_OtherDiagPOA',
  '67D_OtherDiagCode',
  '67D_OtherDiagPOA',
  '67E_OtherDiagCode',
  '67E_OtherDiagPOA',
  '67F_OtherDiagCode',
  '67F_OtherDiagPOA',
  '67G_OtherDiagCode',
  '67G_OtherDiagPOA',
  '67H_OtherDiagCode',
  '67H_OtherDiagPOA',
  '67I_OtherDiagCode',
  '67I_OtherDiagPOA',
  '67J_OtherDiagCode',
  '67J_OtherDiagPOA',
  '67K_OtherDiagCode',
  '67K_OtherDiagPOA',
  '67L_OtherDiagCode',
  '67L_OtherDiagPOA',
  '67M_OtherDiagCode',
  '67M_OtherDiagPOA',
  '67N_OtherDiagCode',
  '67N_OtherDiagPOA',
  '67O_OtherDiagCode',
  '67O_OtherDiagPOA',
  '67P_OtherDiagCode',
  '67P_OtherDiagPOA',
  '67Q_OtherDiagCode',
  '67Q_OtherDiagPOA',
  '66_ICDCode'],
 '68_Label': ['68A_Other', '68B_Other'],
 '69_ADMT_DX': '69_AdmDiagCode',
 '70_PATIENT_REASON_DX': ['70A_PatReasonCode',
  '70B_PatReasonCode',
  '70C_PatReasonCode'],
 '71_ PPS_CODE': '71_PPSCode',
 '72_ECI': ['72A_Ecode', '72B_Ecode', '72C_Ecode'],
 '73_Label': '',
 '74 PRINCIPAL_OTHER_PROCEDURE_CODE': ['74_PrincipleProcCode',
  '74_PrincipleProcDate',
  '74A_OtherProcCode',
  '74A_OtherProcDate',
  '74B_OtherProcCode',
  '74B_OtherProcDate',
  '74C_OtherProcCode',
  '74C_OtherProcDate',
  '74D_OtherProcCode',
  '74D_OtherProcDate',
  '74E_OtherProcCode',
  '74E_OtherProcDate'],
 '75_Label': '',
 '76_ATTENDING_DETAILS': ['76_AttProvFName',
  '76_AttProvLName',
  '76_AttProvNPI',
  '76_AttProvOrgName',
  '76_AttProvOthId',
  '76_AttProvQualId', '76_AttProvFNameDE',
 '76_AttProvFullName',
 '76_AttProvLNameDE'],
 '77_OPERATING_DETAILS': ['77_OpProvFName',
  '77_OpProvLName',
  '77_OpProvNPI',
  '77_OpProvOrgName',
  '77_OpProvOthId',
  '77_OpProvQualId', '77_OpProvFNameDE',
 '77_OpProvFullName',
 '77_OpProvLNameDE'],
 '78_OTHER_DETAILS': ['78_OthProvFName',
  '78_OthProvLName',
  '78_OthProvNPI',
  '78_OthProvNPIQual',
  '78_OthProvOrgName',
  '78_OthProvOthId',
  '78_OthProvQualId', '78_OthProvFNameDE',
 '78_OthProvFullName',
 '78_OthProvLNameDE',
 '78_OthProvNameQual'],
 '79_OTHER_DETAILS': ['79_OthProvFName',
  '79_OthProvLName',
  '79_OthProvNPI',
  '79_OthProvNPIQual',
  '79_OthProvOrgName',
  '79_OthProvOthId',
  '79_OthProvQualId', '79_OthProvFNameDE',
 '79_OthProvFullName',
 '79_OthProvLNameDE',
 '79_OthProvNameQual'],
 '80_REMARKS': ['80_NYSurCharge','MedicaidFlag'],
 '81_Label': ['81A_AddtlCodeValue','81A_AddtlCodeInd',
  '81B_AddtlCodeInd',
  '81B_AddtlCodeValue',
  '81C_AddtlCodeInd',
  '81C_AddtlCodeValue',
  '81D_AddtlCodeInd',
  '81D_AddtlCodeValue']}
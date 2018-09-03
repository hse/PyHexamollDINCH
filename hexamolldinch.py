# Model for HEXAMOLL DINCH Based on Xylene micturition model
# Adding Fetoplacental compartment based on Gaohua et al (2012) Brit J Clin
# Pharmacol 74:5, 873-885
#
from enum import IntEnum
import numpy as np
from scipy.integrate import solve_ivp
import inspect
import math

epsilon = math.sqrt(np.finfo(float).eps)


def is_multiple(n, f):
    '''for checking event timings'''
    return math.isclose(round(n / f) - n / f, 0.0, abs_tol=epsilon)


class P(object):
    '''for a "bunch" of parameters'''

    def __init__(self, dict):
        self.__dict__.update(dict)


def pack_expando(source, target, executing_function):
    '''facilitate easy collection of parameter assignments'''

    # exclude fn args
    formals = inspect.getargs(executing_function)
    args = formals.args

    # assign dict values as obj attributes
    names = list(k for k in source.keys() if k not in args)
    for name in names:
        setattr(target, name, source[name])

    return target


class S(IntEnum):
    '''state indexing (aids readability of derivative code)'''
    ABelly = 0
    ODOSE = 1
    DOSESTEP = 2
    AGiTract = 3
    Agu = 4
    Ast = 5
    Afa = 6
    Arpd = 7
    Aspd = 8
    APreg = 9
    AX = 10
    dose = 11
    AMMINCHU = 12
    AMMINCHB = 13
    VBladder = 14
    binexpinh = 15
    AMgu = 16
    AMliM = 17
    AliM = 18
    APregM = 19
    AfaM = 20
    AspdM = 21
    ArpdM = 22
    ABellyM = 23
    AGiTractM = 24
    AstM = 25
    AguM = 26
    AMli = 27
    Ali = 28
    Aplasm_DINCH = 29
    ARBC_DINCH = 30
    Aplasm_MINCH = 31
    ARBC_MINCH = 32
    AMINCH_Dose = 33
    N_STATE = 34


assert len(S) == S.N_STATE + 1

state_names = [n for n, _ in list(S.__members__.items())[:-1]]
derivative_names = ["R" + n for n in state_names]


def assign_parameters():
    "Simulation constants"

    MWDINCH = 424.67  # DINCH molecular mass [g/mol]
    MWMINCH = 298.41  # MINCH molecular mass [g/mol]
    MWMINCHOH = 314.41  # OH-MINCH molecular mass [g/mol]

    # Mass
    BW0 = 89.  # body mass [kg] 75.12

    # Fractional tissue volumes (%BW) with fraction of blood subtracted
    VT = 0.95  # proportion of vascularised tissue
    VfaC = 0.195  # fractional volume
    VguC = 0.067  # Gut fractional volume
    VstC = 0.0158  # Stomach fractional volume
    VliC = 0.0203  # fractional volume
    VspdC = 0.475  # slowly perfused fractional volume
    VrpdC = 0.033  # rapidly perfused fractional volume
    VpregC = 0.  # fetoplacental fractional volume

    VBldC = 0.079  # blood fractional volume
    VplasC = 0.044  # plasma fractional volume

    FB_DINCH = 0.9998752  # Fraction of DINCH bound to plasma proteins 0.9998752
    FB_MINCH = 0.9854  # Fraction of MINCH bound to plasma proteins 0.9854

    # Caridac output and minute volume
    CAE = 0.75  # Cardiac allometric exponent
    RAE = 0.75  # Respiratory allometric exponent
    QCC = 12.  # Cardiac allometric constant
    QPC = 14.7  # Respiratory allometric constant

    # Fractional blood flows
    QhepartC = 0.06  # hepatic arterial flow
    QfaC = 0.05  # fractional blood flow
    QguC = 0.17  # Gut fractional blood flow
    QstC = 0.01  # Stomach fractional blood flow
    QspdC = 0.29  # overall fractional blood flow to slowly perfused tissue
    QrpdC = 0.42  # overall fractional blood flow to rapidly perfused tissue
    QpregC = 0.  # overall fractional blood flow to fetoplacental compartment

    # Fraction of CYP-mediated metabolism MINCH -> MINCH (Table 2 Koch et al
    FracMetabMOH = 0.8
    # (2013) Arch Tox 87) 0.25

    PORALDOSE = 0.55  # oral dose [mg]
    DRINKTIME = 0.25  # Drink time [h]
    BELLYPERM = 0.685  # [/h]
    GIPERM = 25.1  # [/h]
    KEMAX = 10.2  # [Maximum emptying rate /h]
    KEMIN = 0.005  # [Minimum emptying rate /h]
    KA_MINCH = 0.3  # 1st-order oral uptake rate of MINCH (1/hr)

    # DINCH tissue:blood Partition coefficients(Poulin and Haddad)
    Pbab = 3.01  # Red blood cells:plasma partition coefficient
    Pspdb = 3.29  # Slowly perfused tissue:air partition coefficient
    Plib = 5.89  # tissue:air partition coefficient
    Prpdb = 3.7  # Richly tissue:air partition coefficient
    Pfab = 63.4  # Fat tissue:air partition coefficient
    Pstb = 7.4  # Stomach tissue:air partition coefficient
    Pgub = 7.4  # GI Tract tissue:blood partition coefficient
    Ppregb = 3.7  # fetus:blood partition coefficient

    # MINCH tissue:blood Partition coefficients(Schmitt 2008)
    PbaM = 6.67  # Red blood cells:plasma partition coefficient
    PspdM = 7.51  # Slowly perfused tissue:air partition coefficient
    PliM = 54.8  # tissue:air partition coefficient
    PrpdM = 12.20  # Richly tissue:air partition coefficient
    PfaM = 29.10  # Fat tissue:air partition coefficient
    PstM = 25.2  # Stomach tissue:air partition coefficient
    PguM = 25.2  # GI Tract tissue:blood partition coefficient
    PpregM = 22.7  # fetus:blood partition coefficient

    # Metabolism
    MPY = 34.  # microsomal protein yield [mg microsomal protein/g liver]
    Dinch_half_life = 30.  # Dinch -> Minch half-life (minutes)
    Incub_vol = 1.  # Volume of incubation (ml)
    Microsome_prot = 0.5  # microsomal protein amount (mg)

    Minch_half_life = 30.  # Minch -> OH-Minch and cx-Minch half-life (minutes)

    MPYgu = 20.  # microsomal protein yield [mg microsomal protein/g gut]
    Dinch_GUT_half_life = 30.  # Dinch -> Minch GUT half-life (minutes)

    K1 = 20.  # First-order elimination rate from blood [/h]

    CIppm = 0.  # inhalation exposure concentration [ppm]

    # initialize the concentrations of urine
    # inside the bladder and expelled urine
    RUrine = 0.099  # Rate of Urine Production [l/h]
    Creat = 1.348  # Urinary creatinine concentration [g/L] or 0.01192 [mol/L]

    # Pregnancy related growth constants.

    # Volume of Fetoplacental compartment
    Vprega = 0.01  # Gompertz constant a
    Vpregb = 0.37  # Gompertz constant b
    Vpregc = 0.052  # Gompertz constant c

    # Growth of liver mass coefficients
    Vlia = 0.0000
    Vlib = 0.0000

    # Growth of adipose mass coefficients
    Vfata = 0.007876
    Vfatb = 0.00004667

    # Growth of gut mass coefficients
    Vguta = 0.0000
    Vgutb = 0.0000

    # Growth of blood volume coefficients
    Vblooda = 0.008104375
    Vbloodb = 0.001094423
    Vbloodc = -0.00001824

    # Growth of plasma volume coefficients
    Vplasmaa = 0.00892
    Vplasmab = 0.00168
    Vplasmac = -0.000028

    # Growth of stomach mass coefficients
    Vsta = 0.0000
    Vstb = 0.0000

    # Growth of rapidly perfused mass coefficients
    Vrpda = 0.0000
    Vrpdb = 0.0000

    # Growth of slowly perfused mass coefficients
    Vspda = 0.0000
    Vspdb = 0.0000

    # Fetoplacental compartment: blood flow coefficients
    Qprega = -0.4051  # Coeffient a1
    Qpregb = 0.1188  # Coeffient a2
    Qpregc = -0.0019  # Coeffient a3

    # Growth of hepart blood flow coefficients
    Qheparta = 0.0000
    Qhepartb = 0.0000

    # Growth of adipose blood flow coefficients
    Qfata = 0.01542
    Qfatb = -0.00022

    # Growth of gut blood flow coefficients
    Qguta = 0.0000
    Qgutb = 0.0000

    # Growth of stomach blood flow coefficients
    Qsta = 0.0000
    Qstb = 0.0000

    # Growth of rapidly perfused blood flow coefficients
    # Only the kidney flow changes.  Calculated as a weighted sum of kidney and
    # other flows.
    Qrpda = 0.02146452
    Qrpdb = -0.0005330645

    # Growth of slowly perfused blood flow coefficients
    # Only the skin flow changes.  Calculated as a weigth sum of skin, bone and
    # mucle flows.
    Qspda = 0.005337037
    Qspdb = 0.0000

    # Gestational age when pregnant women exposed to xenobiotics (weeks)
    GA0 = 0

    t_start = 0.
    t_end = 50.    # run time (hr)
    t_int = 0.05  # communication interval

    # Return all variables in this function's environment

    # create a new Parameters obj and assign the above to it
    return P(locals())


def calculate_variables(p):

    BWc = p.BW0 ** p.CAE  # Cardiac scaling output factor (kg)
    BWr = p.BW0 ** p.RAE  # Respiratory scaling output factor (kg)
    QC = p.QCC * BWc
    QP = p.QPC * BWr

    # Gelman reparameterisations
    Qcci = p.QrpdC + p.QspdC + p.QhepartC + p.QfaC + p.QstC + p.QguC + p.QpregC
    Qrpdci = p.QrpdC / Qcci
    Qspdci = p.QspdC / Qcci
    Qhepartci = p.QhepartC / Qcci
    Qfaci = p.QfaC / Qcci
    Qstci = p.QstC / Qcci
    Qguci = p.QguC / Qcci

    HEME = 1 - (p.VplasC / p.VBldC)  # Volume of Haeme
    VRBC = HEME * p.VBldC  # Volume of red blood cells

    Vti = (1 - p.VT) + p.VrpdC + p.VspdC + p.VliC + p.VfaC + \
        p.VstC + p.VguC + p.VpregC + p.VplasC + VRBC
    Vguci = p.VguC / Vti
    Vstci = p.VstC / Vti
    Vfaci = p.VfaC / Vti
    Vlici = p.VliC / Vti
    Vspdci = p.VspdC / Vti
    Vrpdci = p.VrpdC / Vti
    Vbldci = p.VBldC / Vti
    Vplasci = p.VplasC / Vti
    VRBCci = VRBC / Vti

    Vli0 = Vlici * p.BW0  # scaled liver fractional volume
    Vfa0 = Vfaci * p.BW0  # scaled adipose fractional volume
    Vgu0 = Vguci * p.BW0  # scaled gut fractional volume
    Vst0 = Vstci * p.BW0  # scaled stomach fractional volume
    Vrpd0 = Vrpdci * p.BW0  # scaled rapildy perfused tissue fractional volume
    Vspd0 = Vspdci * p.BW0  # scaled slowly perfused tissue fractional volume
    Vbld0 = Vbldci * p.BW0  # scaled blood tissue fractional volume
    # unbound model
    Vplas0 = Vplasci * p.BW0  # scaled blood plasma fractional volume
    VRBC0 = VRBCci * p.BW0

    # unbound model

    Qrpd0 = Qrpdci * QC  # scaled rapildy perfused tissue fractional blood flow
    Qspd0 = Qspdci * QC  # scaled slowly perfused tissuefractional blood flow
    Qhepart0 = Qhepartci * QC  # scaled fractional blood flow
    Qfa0 = Qfaci * QC  # scaled adipose fractional blood flow
    Qgu0 = Qguci * QC  # scaled gut fractional blood flow
    Qst0 = Qstci * QC  # scaled stomach fractional blood flow

    ORALDOSE = p.PORALDOSE * p.BW0
    DOSEFLOW = ORALDOSE / p.DRINKTIME

    # Initial conditions for model
    ABelly0 = 0.0
    ODOSE0 = 0.0
    AGiTract0 = 0.0
    Agu0 = 0.0
    Ast0 = 0.0
    Afa0 = 0.0
    Arpd0 = 0.0
    Aspd0 = 0.0
    APreg0 = 0.0
    AX0 = 0.0
    dose0 = 0.0
    AMMINCHU0 = 0.0
    AMMINCHB0 = 0.0
    VBladder0 = 0.0
    AMgu0 = 0.0
    AMliM0 = 0.0
    AliM0 = 0.0
    APregM0 = 0.0
    AfaM0 = 0.0
    AspdM0 = 0.0
    ArpdM0 = 0.0
    ABellyM0 = 0.0
    AGiTractM0 = 0.0
    AstM0 = 0.0
    AguM0 = 0.0
    AMli0 = 0.0
    Ali0 = 0.0
    Aplasm_DINCH0 = 0.0  # unbound model
    ARBC_DINCH0 = 0.0  # unbound model
    Aplasm_MINCH0 = 0.0  # unbound model
    ARBC_MINCH0 = 0.0  # unbound model
    AMINCH_Dose0 = 0.0

    pack_expando(locals(), p, inspect.currentframe().f_code)


def est_gest_flow(Qpreg, GA):

    # Edit by K McNally 15/02/18
    # Piecewise linear fit to ensure positive flow between zero and four weeks

    # use scipy.stats.linregress here?  do me a favour

    return np.where(GA >= 4., Qpreg, GA * 0.1588 / 4.)


def derivative_impl(y, p, GA):

    Vli = p.Vli0 * (1 + p.Vlia * GA + p.Vlib * GA ** 2)
    Vgu = p.Vgu0 * (1 + p.Vguta * GA + p.Vgutb * GA ** 2)
    Vst = p.Vst0 * (1 + p.Vsta * GA + p.Vstb * GA ** 2)
    Vfa = p.Vfa0 * (1 + p.Vfata * GA + p.Vfatb * GA ** 2)
    Vrpd = p.Vrpd0 * (1 + p.Vrpda * GA + p.Vrpdb * GA ** 2)
    Vspd = p.Vspd0 * (1 + p.Vspda * GA + p.Vspdb * GA ** 2)
    Vbld = p.Vbld0 * (1 + p.Vblooda * GA + p.Vbloodb *
                      GA ** 2 + p.Vbloodc * GA ** 3)
    Vplas = p.Vplas0 * (1 + p.Vplasmaa * GA + p.Vplasmab *
                        GA ** 2 + p.Vplasmac * GA ** 3)
    Vpreg = p.Vprega * np.exp((p.Vpregb / p.Vpregc) *
                              (1 - np.exp(-p.Vpregc * GA)))

    # Calculate VRBC from VRBC0 and fix Vtotal
    Vtotal = Vli + Vgu + Vst + Vfa + Vrpd + Vspd + p.VRBC0 + Vplas + Vpreg

    Qhepart = p.Qhepart0 * (1 + p.Qheparta * GA + p.Qhepartb * GA ** 2)
    Qgu = p.Qgu0 * (1 + p.Qguta * GA + p.Qgutb * GA ** 2)
    Qst = p.Qst0 * (1 + p.Qsta * GA + p.Qstb * GA ** 2)
    Qli = Qgu + Qst + Qhepart
    Qfa = p.Qfa0 * (1 + p.Qfata * GA + p.Qfatb * GA ** 2)
    Qrpd = p.Qrpd0 * (1 + p.Qrpda * GA + p.Qrpdb * GA ** 2)
    Qspd = p.Qspd0 * (1 + p.Qspda * GA + p.Qspdb * GA ** 2)
    Qpreg = 1 * (0 + p.Qprega * GA + p.Qpregb * GA ** 2 + p.Qpregc * GA ** 3)

    Qpreg = est_gest_flow(Qpreg, GA)

    QCMC = Qhepart + Qgu + Qst + Qfa + Qrpd + Qspd + Qpreg

    # ********************************************************************************************!
    #                  DINCH Concentrations in Compartments
    # ********************************************************************************************!

    RODOSE = p.DOSEFLOW * y[S.DOSESTEP]  # amount absorbed (mg)
    IH = p.CIppm * y[S.binexpinh] * \
        (p.MWDINCH / 24450)  # inhalation infusion (mg/L)

    CPreg = y[S.APreg] / Vpreg  # Fetoplacental cellular concentration (mg/L)
    Cgu = y[S.Agu] / Vgu  # gut concentration (mg/L)
    Cst = y[S.Ast] / Vst  # concentration in stomach (mg/L)
    Cfa = y[S.Afa] / Vfa  # concentration in fat (mg/L)
    Cli = y[S.Ali] / Vli  # concentration in liver (mg/L)
    Cspd = y[S.Aspd] / Vspd  # concentration in slowly perfused tissue (mg/L)
    Crpd = y[S.Arpd] / Vrpd  # concentration in richly perfused tissue(mg/L)

    GPER = p.KEMAX / (1 + p.KEMIN * Cst)

    # Fetoplacental organ concentration concentration (mg/L)
    CVPreg = CPreg / p.Ppregb
    CVgu = Cgu / p.Pgub  # gut venous organ concentration kidney (mg/L)
    CVst = Cst / p.Pstb  # stomach venous organ concentration (mg/L)
    CVfa = Cfa / p.Pfab  # venous concentration fat (mg/L)
    CVli = Cli / p.Plib  # venous concentration liver (mg/L)
    # venous organ concentration slowly perfused tissue (mg/L)
    CVspd = Cspd / p.Pspdb
    # venous organ concentration richly perfused tissue (mg/L)
    CVrpd = Crpd / p.Prpdb

    # Blood compartment broken down to plasma and red blood cells and plasma
    # compartment
    # unbound model
    Aplasmub_DINCH = y[S.Aplasm_DINCH] * (1 - p.FB_DINCH)  # Fraction unbound
    CA_DINCH = Aplasmub_DINCH / Vplas  # Arterial concentration (mg/L)
    # Concentration in red blood cells (mg/L)
    CRBC_DINCH = y[S.ARBC_DINCH] / p.VRBC0
    # unbound model

    # ********************************************************************************************!
    #                  DINCH Differential Equations
    # ********************************************************************************************!

    Vli = p.Vli0 * (1 + p.Vlia * GA + p.Vlib * GA ** 2)

    # Clearance (L/h whole liver)
    Clintdinch = (0.693 / p.Dinch_half_life) * \
        (p.Incub_vol / p.Microsome_prot) * p.MPY * Vli * 60

    # Clearance (L/h whole liver)
    Clintminch = (0.693 / p.Minch_half_life) * \
        (p.Incub_vol / p.Microsome_prot) * p.MPY * Vli * 60

    # Clearance (L/h whole liver)
    Clintdinchgu = (0.693 / p.Dinch_GUT_half_life) * \
        (p.Incub_vol / p.Microsome_prot) * p.MPYgu * Vgu * 60

    # DINCH venous concentration (mg/L)
    CV = ((CVPreg * Qpreg) + (CVspd * Qspd) +
          (CVrpd * Qrpd) + (CVfa * Qfa) + (CVli * Qli)) / QCMC

    # Blood compartment broken down to plasma and red blood cells and plasma
    # compartment
    # unbound model
    # Rate of change in amount in red blood cells
    RARBC_DINCH = (CA_DINCH - CRBC_DINCH / p.Pbab)
    # Rate of change in amount Bound + Unbound
    RAplasm_DINCH = QCMC * (CV - CA_DINCH) - RARBC_DINCH + (p.QP * IH)
    CX = CA_DINCH / 10000000000000000
    # unbound model

    # DINCH fetoplacental compartment derivative (mg/h/kg)
    RAPreg = Qpreg * (CA_DINCH - CVPreg)
    RAMgu = ((Qgu * Clintdinchgu) / (Qgu + Clintdinchgu / p.Pbab)) * CVgu

    # DINCH rate of uptake in stomach compartment (mg/h/kg)
    RABelly = (RODOSE) - (GPER * y[S.ABelly]) - (p.BELLYPERM * y[S.ABelly])
    # DINCH rate of uptake in GI Tract compartment (mg/h/kg)
    RAGiTract = (GPER * y[S.ABelly]) - (p.GIPERM * y[S.AGiTract])
    # DINCH rate of change in STOMACH compartment (mg/h/kg)
    RAst = Qst * (CA_DINCH - CVst) + p.BELLYPERM * y[S.ABelly]
    # DINCH rate of change in GUT compartment (mg/h/kg)
    RAgu = Qgu * (CA_DINCH - CVgu) + p.GIPERM * y[S.AGiTract] - RAMgu
    # DINCH fat compartment derivative (mg/h/kg)
    RAfa = Qfa * (CA_DINCH - CVfa)

    RAMli = ((Qli * Clintdinch) / (Qli + Clintdinch / p.Pbab)) * \
        CVli  # DINCH rate of change of metabolism (mg/h/kg)

    RAli = (Qhepart * CA_DINCH) + (Qst * CVst) + \
        (Qgu * CVgu) - (Qli * CVli) - RAMli

    # DINCH slowly perfused compartment derivative (mg/h/kg)
    RAspd = Qspd * (CA_DINCH - CVspd)
    # DINCH richly perfused compartment derivative (mg/h/kg)
    RArpd = Qrpd * (CA_DINCH - CVrpd)
    RAX = p.QP * CX  # DINCH amount exhaled derivative (mg)
    Rdose = p.QP * IH  # DINCH dose derivative (mg)

    # ********************************************************************************************!
    #                  MINCH Concentrations in Compartments
    # ********************************************************************************************!
    CliM = y[S.AliM] / Vli  # MINCH concentration in liver (mg/L)
    CVliM = CliM / p.PliM  # MINCH venous concentration liver (mg/L)

    # MINCH concentration in fetoplacental compartment (mg/L)
    CPregM = y[S.APregM] / Vpreg
    # MINCH venous concentration in fetoplacental compartment (mg/L)
    CVPregM = CPregM / p.PpregM

    CfaM = y[S.AfaM] / Vfa  # MINCH venous concentration in fat (mg/L)
    CVfaM = CfaM / p.PfaM  # MINCH venous concentration fat (mg/L)

    # MINCH concentration in slowly perfused tissue (mg/L)
    CspdM = y[S.AspdM] / Vspd
    # MINCH venous organ concentration slowly perfused tissue (mg/L)
    CVspdM = CspdM / p.PspdM

    # MINCH concentration in richly perfused tissue(mg/L)
    CrpdM = y[S.ArpdM] / (Vrpd + Vgu + Vst)
    # MINCH venous organ concentration richly perfused tissue (mg/L)
    CVrpdM = CrpdM / p.PrpdM

    # Gut and stomach masses absorbed into rpd compartment
    CguM = y[S.AguM] / Vgu  # MINCH concentration in gut (mg/L)
    CVguM = CguM / p.PguM  # MINCH gut venous organ concentration kidney (mg/L)

    # unbound model
    Aplasmub_MINCH = y[S.Aplasm_MINCH] * (1 - p.FB_MINCH)
    CA_MINCH = Aplasmub_MINCH / Vplas
    CRBC_MINCH = y[S.ARBC_MINCH] / p.VRBC0
    # unbound model

    # ********************************************************************************************!
    #                  MINCH Differential Equations
    # ********************************************************************************************!

    # Edits on 20/07/2018 by K.McNally
    # Changes made here to correct the source term for MINCH
    # Need MINCH to be created in the gut and liver at a rate that's equal to the
    # elimination of DINCH
    # Changes to the delivery of MINCH in the gut compared with DINCH.
    # Instantaneous appearence in the gut as opposed to absorption through GI
    # tract.

    # venous concentration (mg/L)
    CVM = ((CVliM * Qli) + (CVPregM * Qpreg) + (CVfaM * Qfa) +
           (CVspdM * Qspd) + (CVrpdM * Qrpd)) / QCMC

    # unbound model
    # Blood compartment broken down to plasma and red blood cells and plasma
    # compartment
    RARBC_MINCH = (CA_MINCH - CRBC_MINCH / p.PbaM)
    RAplasm_MINCH = QCMC * (CVM - CA_MINCH) - RARBC_MINCH
    # unbound model

    RABellyM = 0
    RAGiTractM = 0
    RAstM = 0
    RAguM = 0

    RAMliM = ((Qli * Clintminch) / (Qli + Clintminch / p.PbaM)) * CVliM

    # Simplified model for the liver for MINCH.
    # Two source terms for MINCH (from gut and liver appear in the liver)
    # Last terms appears through metabolism of DINCH in liver
    RAliM = Qli * (CA_MINCH - CVliM) - (RAMliM) + RAMli + RAMgu

    RAMINCH_Dose = Qli * (CA_MINCH - CVliM) - (RAMliM)

    # MINCH pregnancy compartment derivative (mg/h/kg)
    RAPregM = Qpreg * (CA_MINCH - CVPregM)
    # MINCH fat compartment derivative (mg/h/kg)
    RAfaM = Qfa * (CA_MINCH - CVfaM)
    # MINCH slowly perfused compartment derivative (mg/h/kg)
    RAspdM = Qspd * (CA_MINCH - CVspdM)
    # MINCH richly perfused compartment derivative (mg/h/kg)
    RArpdM = Qrpd * (CA_MINCH - CVrpdM)

    # ********************************************************************************************!
    #                  MINCH Urinary excretion
    # ********************************************************************************************!

    RAMMINCHB = (RAMliM * p.FracMetabMOH) * (p.MWMINCHOH / p.MWMINCH) - \
        (p.K1 * y[S.AMMINCHB])  # amount of MINCH in blood (mg)

    # amount of MINCH in bladder compartment (mg)
    RAMMINCHU = p.K1 * y[S.AMMINCHB]

    return locals()


def derivative(t, y, p):

    # compute on time here
    GA = p.GA0 + (t / (24 * 7))

    # proceed with (non-vectorized) calculation
    derivatives_plus = derivative_impl(y, p, GA)

    # solver needs derivatives transferred into array
    dy = [derivatives_plus[derivative_names[i]]
          if derivative_names[i] in derivatives_plus else 0.
          for i in range(len(derivative_names))]

    dy[S.VBladder] = p.RUrine

    return dy


def compute_outputs_impl(y, p, v):

    # MINCH Venous concentration (micromoles/L)
    CVMumol = (v["CVM"] / p.MWMINCH) * 1000
    CVumol = (v["CV"] / p.MWDINCH) * 1000
    CAT_DINCH = y[S.Aplasm_DINCH] / v["Vplas"]
    CAT_MINCH = y[S.Aplasm_MINCH] / v["Vplas"]
    AUCREmmol = y[S.AMMINCHU]  # (mg)

    # this code ought to avoid dividing by zero...
    # creat = y[S.VBladder] * p.Creat
    # Curine = np.where(creat == 0., 0., AUCREmmol / creat)

    # ...  but it doesn't so we'll do it this way
    Curine = np.zeros(len(AUCREmmol))
    for i in range(len(AUCREmmol)):
        creat = y[S.VBladder][i] * p.Creat
        if creat != 0.:
            Curine[i] = AUCREmmol[i] / creat

    # mass in system (mg)
    mass = (y[S.Aspd] + y[S.Arpd] + y[S.Afa] + y[S.Ali] + y[S.AMli] + y[S.Ast] +
            y[S.Agu] + y[S.AMgu] + y[S.APreg] + y[S.Aplasm_DINCH] + y[S.ARBC_DINCH])

    # mass in system (mg)
    # last three terms in this next equation are the amount of un-metabolised
    massM = (y[S.AspdM] + y[S.ArpdM] + y[S.AfaM] + y[S.AliM] + y[S.AMliM] + y[S.AstM] + y[S.AguM] +
             y[S.APregM] + y[S.Aplasm_MINCH] + y[S.ARBC_MINCH] + (mass - y[S.AMgu] - y[S.AMli]))

    relinh = mass / (y[S.dose] + 1e-10)
    reloral = mass / (y[S.ODOSE] + 1e-10)
    relboth = mass / (y[S.dose] + y[S.ODOSE] + 1e-10)
    rel_MINCH = massM / (y[S.AMINCH_Dose] + 1e-10)

    outputs = locals()

    # don't return fn args and non-pbpk variables
    args = inspect.signature(compute_outputs_impl).parameters.keys()
    for arg in args:
        del outputs[arg]
    del outputs["i"]

    return outputs


def compute_outputs(t, y, p):

    # re-run derivative computation using vectorized state
    GA = p.GA0 + (t / (24 * 7))
    recomputed = derivative_impl(y, p, GA)
    outputs = compute_outputs_impl(y, p, recomputed)

    return recomputed, outputs


def pbpk(p, s):

    calculate_variables(p)

    # Solve ODE system
    method = "Radau"  # we're stiff?

    # set up and initialize system state
    y0 = np.zeros(S.N_STATE)

    for state_name, state_value in list(S.__members__.items())[:-1]:
        y0[state_value] = getattr(p, state_name + "0", 0.)

    # initialize scheduling
    t_start = p.t_start

    # if the schedule contains starting values, apply them to initial state then
    # remove from schedule
    tp0 = [tp for tp in s if tp[0] == t_start]
    if any(tp0):
        _, sc = tp0[0]
        for i, v in sc:
            y0[i] = v
        s = s[1:]

    # thunk to pass parameters via scipy routine
    def fun(t, y): return derivative(t, y, p)

    # accumulate solution data in arrays
    y = np.empty(shape=(S.N_STATE, 0))
    t = []
    nfev = 0
    njev = 0
    nlu = 0

    # run scheduling
    for t_end, state_changes in s:

        assert is_multiple(
            t_end, p.t_int), f"Timepoint {t_end} is not a multiple of period {p.t_int}"
        t_eval = np.arange(t_start, t_end + epsilon, p.t_int)
        solution = solve_ivp(
            fun, (t_eval[0], t_eval[-1]), y0, method=method, t_eval=t_eval)
        y = np.append(y, solution.y[:, :-1], axis=1)
        t = np.append(t, solution.t[:-1])
        nfev += solution.nfev
        njev += solution.njev
        nlu += solution.nlu

        y0 = solution.y[:, -1]
        for i, v in state_changes:
            y0[i] = v
        t_start = t_end

    if t_start < p.t_end:
        # run to end
        t_eval = np.arange(t_start, p.t_end + epsilon, p.t_int)
        solution = solve_ivp(
            fun, (t_eval[0], t_eval[-1]), y0, method=method, t_eval=t_eval)
        y = np.append(y, solution.y, axis=1)
        t = np.append(t, solution.t)
        nfev += solution.nfev
        njev += solution.njev
        nlu += solution.nlu

    recomputed, outputs = compute_outputs(t, y, p)

    return t, y, recomputed, outputs


def get_default_schedule():

    schedule = [
        #    t   (state, value), ...
        # ====== =====================
        (0.00, [(S.DOSESTEP, 1), (S.binexpinh, 1),
                (S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (0.25, [(S.DOSESTEP, 0)]),
        (0.80, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (1.50, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (2.30, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (3.00, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (3.80, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (4.00, [(S.binexpinh, 0)]),
        (5.00, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (6.30, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (7.90, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (9.90, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (12.2, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (14.7, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (18.3, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (21.9, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (23.9, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (26.4, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (29.1, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (31.4, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (35.3, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (45.5, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]),
        (48.0, [(S.VBladder, 0.), (S.AMMINCHU, 0.)]), ]

    return schedule


def main():

    # set up, run, and check...
    parameters = assign_parameters()
    schedule = get_default_schedule()

    t, y, r, o = pbpk(parameters, schedule)

    # spot check against R output (allowing 0.1% tolerance)
    assert math.isclose(y[S.ABelly, 25], 3.442968e-04, rel_tol=0.001)
    assert math.isclose(r["CVli"][131], 7.744438e-04, rel_tol=0.001)
    assert math.isclose(r["CPreg"][539], 7.455110e-04, rel_tol=0.001)
    assert math.isclose(o["Curine"][500], 1.163521, rel_tol=0.001)

    return t, y, r, o


def benchmark():

    import timeit
    nRuns = 10
    secs = timeit.timeit(
        "_ = pbpk(parameters, schedule)",
        number=nRuns,
        globals=globals()
    )
    print(f"{nRuns} runs took {secs:.3}s")

    # Notes
    # 2060 - no. of times scipy.integrate.solve_ivp(Radau) calls derivative
    # 2306 - no. of times deSolve::ode() calls derivative in R

    # 4.2 secs - time taken to run sim 10 times
    # 21 secs - time taken to run R implementation 10 times


def plot_trace():
    import matplotlib.pyplot as plt
    which_curine_non_zero = np.where(o["Curine"] != 0.)
    plt.plot(t[which_curine_non_zero][0:199], o["Curine"]
             [which_curine_non_zero][0:199], label="Curine")
    plt.xlabel("time")
    plt.ylabel("conc")
    plt.title("Trace")
    plt.legend()
    plt.show()


def morris():
    from SALib.analyze import morris
    from SALib.sample.morris import sample

    problem = {
        'num_vars': 5,
        'names': ['HEME', 'K1', 'MPY', 'MPYgu', 'QCC'],
        'groups': None,
        'bounds': [[.4, .5],
                   [10., 30.],
                   [30., 40.],
                   [15., 25.],
                   [10., 15.]]
    }

    parameter_values = sample(
        problem,
        N=10,
        num_levels=4,
        grid_jump=2,
        optimal_trajectories=None
    )

    parameters = assign_parameters()
    schedule = get_default_schedule()

    y_index = np.argmax(o["Curine"])

    ys = []

    for inputs in parameter_values:
        for i in range(len(problem["names"])):
            setattr(parameters, problem["names"][i], inputs[i])

        _, __, ___, output = pbpk(parameters, schedule)

        ys.append(output["Curine"][y_index])

    sis = morris.analyze(
        problem,
        parameter_values,
        np.array(ys),
        conf_level=0.95,
        print_to_console=True,
        num_levels=4,
        grid_jump=2,
        num_resamples=100
    )

    import matplotlib.pyplot as plt
    from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms

    fig, (ax1, ax2) = plt.subplots(1, 2)
    horizontal_bar_plot(ax1, sis, {}, sortby='mu_star', unit=r"mg/g")
    covariance_plot(ax2, sis, {}, unit=r"mg/g")

    fig2 = plt.figure()
    sample_histograms(fig2, parameter_values, problem, {'color': 'y'})
    plt.show()


if __name__ == "__main__":
    t, y, r, o = main()
else:
    pass

# EasyGSHpy


from main_tool import j_all

j_all(lib_func_fol , commonfol , basefol , stationsDB , requiredStationsFile , slffile , reqvar , telmod ,  period_s , period_e , k , *args ):
    
    This tool serves to:
        extract required variables from a .slf file (tomawac or 2d modules) for required stations (coordinates)
        and then run a comparison between the extracted values (simulated values) and measured values for:
            water levels
            tides components (amplitude phase shift) after extracting them
            salinity
            velocity components (velocity u, velocity v, velocity magnitude and velocity direction)
            wave components (wave height swh , wave direction mwd, wave mean period mwp and wave peak period pwp)
        
        Parameters
        ----------
        lib_func_fol : folder directory
            Folder location which contains 'gshtools' and 'pputils' sub-folders.
        commonfol : folder directory
            Common folder location. This folder should contain 'measurements' sub-folder containing measured
            values of different variables amongst different stations (yyyy.cu.dat , yyyy.sa.dat , yyyy.wl.dat
             and yyyy.wv.dat)
        basefol: folder directory
            Base folder location. An empty folder where the results are stored.
        stationsDB : .dat file directory 
            File containing info about all stations like names, easting, northing, latitudes and numbering(this 
            numbering starts from France and ends up in England going counter clickwise around the North Sea)
        requiredStationsFile : .dat file directory
            File contaning the names of stations to be compared.
        slffile : .slf file directory 
            Selafin file directory.
        reqvar : array_like
            List of variables abreviations to be extracted from the .slf file (referr to TelDict.py for abrv)
            example : reqvar = ['U' , 'V' , 'S' , 'SLNT' , 'W' ,  'A' , 'G' , 'H']
        telmod : string
            Selafin module. '2D' for 2d module and 'TOMAWAC' for tomawac module.
        period_s : string
            Comparison starting period. example : period_s = '2015-01-06' 
        period_e : string
            Comparison ending period. example : period_e = '2015-01-06' 
        k : int
            Used for water level moving average calculations. k=147 default
        
        *args
        -----
        'a' : extract variables function
        'b' : water level comparison 
        'c' : extract partial tide coefficients and compare them
        'd' : salinity comparison
        'e' : velocity components comparison
        'f' : wave components coomparison
        
        returns
        -------
        in commfol : 'coef_measured' folder containing partial tides coeffitient for measured data
        in basefol :
            'telemac_variables' folder containing simulated values of extracted variables
            'wlcomp' folder containing .png files of water level comparison. for each station:
                -water level comparison and water level difference
                -water level density scatter plot
            for all stations:
                -NRMSE of water level
                -RMSE, MEA and ME of water level
            'ptcomp' folder containing files of partial tides comparison. for each tide:
                -amplitude comparison of required stations
                -phase shift comparison of required stations
            'coef_simulated' folder containing partial tides coeffitient for simulated data
            'salinitycomp' containing files of salinity comparison
            'velocitycomp' containing files of velocity comparison. for each station:
                -velocity u comparison
                -velocity v comparison
                -velocity direction comparison
                -velocity magnitude comparison
            'wavecomp' containing files of wave comparison. for each station:
                -wave height comparison
                -wave direction comparison
                -wave mean period comparison
                -wave peak period comparison
    

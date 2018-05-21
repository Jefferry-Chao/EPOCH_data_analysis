# -*- coding: utf-8 -*-
#*************************************************************************
#***File Name: epoch_class.py
#***Author: Zhonghai Zhao
#***Mail: zhaozhonghi@126.com 
#***Created Time: 2018年03月25日 星期日 14时39分05秒
#*************************************************************************
class epoch_class(object):
    '''
    This class contanin some functions to visulize sdf data file.
    '''
#initialization
    def __init__(self):
        pass
#line plot parameter
    def line_set(self):
        '''
        This function stored some predefined line plot parameters.
        '''
        col_sty = ['r-','g--','b-.','c:','k-','y-','m-']
        #col_sty = ['r-','b-.','c:','k-','y-','m-']
        return col_sty
#get file name list
    def get_list(self,filename):
        '''
        This function is used to get file name list.
        parameters:
        filename----filenames,integer or integer list.
        '''
        import sys
        if(type(filename) == type(1)):
            return [filename]
        elif(type(filename) == type([1,2])):
            return filename
        else:
            print 'File error!'
#get data from a sdf file
    def get_data(self,filename,prefix='1'):
        '''
        This function is used to get the data dictionary from a sdf file.
 e       parameters:
        filename----a sdf file name is needed, defualt:0001.sdf
        prefix------file name prefix, default:1
        '''
        import sdf
        #construct file name
        if(prefix != 'None'):
            sdfname = prefix+str(filename).zfill(4)+'.sdf'
        else:
            sdfname = str(filename).zfill(4)+'.sdf'
        sdfdata = sdf.read(sdfname)
        data_dict = sdfdata.__dict__
        return data_dict
#use a loop to get the exact physical field
    def get_field(self,field='bx',keys=["Magnetic_Field_Bx"]):
        '''
        This function is used to get the exact physical field name in keys.
        parameters:
        field-------the field name. default:'bx'
        keys--------the field names list, default:"Magnetic_Field_Bx"
        '''
        import sys
        ifornot = False
        for eachone in keys:
            if (field.lower() in eachone.lower()):
                field = eachone
                ifornot = True
                break
        if(ifornot == False):
            print "There is not such a field name %s."%(field)
            #sys.exit(0)
        return field
#use a dictionary to chose the normalization constant
    def get_constant(self,field='Magnetic_Field_Bx'):
        '''
        This function is used to chose the normalization constant according to the field.
        parameters:
        field-------the field name. default:'Magnetic_Field_Bx'
        '''
        from constants import bubble_mr as const
        normal = {"magnetic":const.B0,"electric":const.E0,\
                  "current":const.J0,"density":const.N0,\
                  "temperature":const.T0,"ekbar":const.T0,\
                  "xz_u":const.V0,"xz_t":const.T0,"axis":const.D0,\
                  "derived_j":const.J0}
        normal_s = normal.keys()
        for eachone in normal_s:
            if(eachone in field.lower()):
                factor = normal[eachone]
                break
        return factor
#if magnitude is True, this function will return a vector's module
    def get_module(self,data_dict,field='magnetic',gf=False,g_field=150):
        '''
        This function is used to calculate a vector's module.
        parameters:
        data_dict---a data dictionary read from a sdf file. 
        field-------a vector field, default:magnetic
        gf----------guide field, default:False
        g_field-----guide field, default:150
        '''
        import numpy as np
        mode = 0
        const = 0
        s = data_dict.keys()
        for eachone in s:
            if((field.lower() in eachone.lower()) and \
              ('rank' not in eachone.lower())):
                const = const + 1
                data_field = data_dict[eachone]
                array = data_field.data
                if((gf == True) and (eachone == 'Magnetic_Field_Bz')):
                    array = array - g_field
                mode = mode + array*array
        array = np.sqrt(mode)
        if(const < 3):
            print "Warning! There is not enough componens in this vector!"
        return np.transpose(array)
#get line index
    def get_index(self,data_dict,field='Magnetic_Field_Bx',axes='x'):
        '''
        This function is used to get line data from an array.
        parameters:
        data_dict---data dictionary from a sdf file.
        field-------physical field, default:'Magnetic_Field_Bx'
        axes--------axes,'x' or 'y', default:'x'
        '''
        import numpy as np
        field_data = data_dict[field]
        data = field_data.data
        array = np.transpose(data)
        shape = array.shape
        if (axes == 'x'):
            print "There are %d rows in this field array, please chose proper one."%(shape[0])
        else:
            print "There are %d cloumns in this field array, please chose proper one."%(shape[1])
        index = raw_input("Please input a proper integer number:")
        index = int(index)
        return index
#get line array
    def get_line(self,data_dict,field='Magnetic_Field_Bx',axes='x',index=1):
        '''
        This function is used to get line data from an array.
        parameters:
        data_dict---data dictionary from a sdf file.
        field-------physical field, default:'Magnetic_Field_Bx'
        axes--------axes,'x' or 'y', default:'x'
        index-------array index, default:0
        '''
        import numpy as np
        field_data = data_dict[field]
        data = field_data.data
        array = np.transpose(data)
        if (axes == 'x'):
            vector = array[index-1,:]
        else:
            vector = array[:,index-1]
        return vector
#get module line
    def get_module_line(self,array,axes='x',index=0):
        '''
        This function is  used to get a line from an array.
        parameters:
        array-------input array
        axes--------axes,'x' or 'y', default:'x'
        index-------array index, default:0
        '''
        if (axes == 'x'):
            vector = array[index-1,:]
        else:
            vector = array[:,index-1]
        return vector
#get extent
    def get_extent(self,data_dict):
        '''
        This function is used to get array extent.
        parameters:
        data_dict---data dictionary read from sdf file.
        '''
        from constants import bubble_mr as const
        #import numpy as np
        grid = data_dict["Grid_Grid_mid"]
        #use this method, sometimes contour core dumped
        #grid = data_grid.data
        #x = grid[0]/const.di
        #y = grid[1]/const.di
        #extent = [np.min(x),np.max(x),np.min(y),np.max(y)]
        grid_extent = grid.extents
        xmin = grid_extent[0]/const.D0
        xmax = grid_extent[2]/const.D0
        ymin = grid_extent[1]/const.D0
        ymax = grid_extent[3]/const.D0
        extent = [xmin,xmax,ymin,ymax]
        return extent
#get array
    def get_array(self,data_dict,field='Magnetic_Field_Bx',gf=False,g_field=150):
        '''
        This function is used to get array frim a sdf dictionary.
        parameters:
        data_dict---data dictionary.
        field-------field, default:'Magnetic_Field_Bx'
        gf----------guide field, default:False
        g_field-----guide field, default:150
        '''
        import numpy as np
        data_array = data_dict[field]
        array = data_array.data
        array = np.transpose(array)
        if((gf == True) and (field == 'Magnetic_Field_Bz')):
            array = array - g_field
        return array
#get cordinate
    def get_axis(self,data_dict,axes='x'):
        '''
        This function is used to get 'x', 'y' cordinate.
        parameters:
        data_dict---data dictionary.
        axes--------axes, 'x' or 'y', default:'x'
        '''
        import numpy as np
        grid = data_dict["Grid_Grid_mid"]
        dimens = grid.dims
        extent = grid.extents
        if(axes == 'x'):
            axis = np.linspace(extent[0],extent[2],dimens[0])
        else:
            axis = np.linspace(extent[1],extent[3],dimens[1])
        return axis
#get cordinate 3d
    def get_axis_3d(self,data_dict,axes='x'):
        '''
        This function is used to get 'x', 'y' cordinate.
        parameters:
        data_dict---data dictionary.
        axes--------axes, 'x' or 'y' or 'z', default:'x'
        '''
        import numpy as np
        grid = data_dict["Grid_Grid_mid"]
        dimens = grid.dims
        extent = grid.extents
        if(axes == 'x'):
            axis = np.linspace(extent[0],extent[3],dimens[0])
        elif(axes == 'y'):
            axis = np.linspace(extent[1],extent[4],dimens[1])
        else:
            axis = np.linspace(extent[2],extent[5],dimens[2])
        return axis
#line integrate
    def line_integrate(self,namelist,field='bx',axes='y',magnitude=False,semi=True,\
                       max_or_min=False,prefix='1'):
        '''
        Thismagnitude---in integrate a vector's module, set True, default:False
 function is used to integrate field along a line,'x' or 'y'.
        parameters:
        namelist----sdf name list.
        field-------physical field to be integrated, default:'bx'
        axes--------axes, 'x' or 'y' ,default:'y'.
        magnitude---in integrate a vector's module, set True, default:False
        semi--------if integrate semi-axis, set True, default:True
        max_or_min--when semi is True, if find extreme min, set False, else set True.
        prefix------file name prefix, default:1
        '''
        import numpy as np
        from constants import bubble_mr as const
        #use sample sdfget parameters
        data_dict = epoch_class.get_data(self,namelist[0],prefix=prefix)
        keys = data_dict.keys()
        if(magnitude == False):
            field = epoch_class.get_field(self,field=field,keys=keys)
        else:
            field_d = field
            field = epoch_class.get_field(self,field=field,keys=keys)
        data = np.array((data_dict[field]).data)
        dimen = data.shape
        a = dimen[1]
        b = dimen[0]
        axis = epoch_class.get_axis(self,data_dict,axes=axes)
        dx = axis[1]-axis[0]
        constant = epoch_class.get_constant(self,field=field)
        n = len(namelist)
        integrate = np.zeros(n)
        #use a loop to integrate filed
        for i in range(n):
            data_dict = epoch_class.get_data(self,namelist[i],prefix=prefix)
            if(magnitude == False):
                array = epoch_class.get_array(self,data_dict,field=field)
            else:
                array = epoch_class.get_module(self,data_dict,field=field_d)
            if(axes == 'x'):
                vector = (array[a/2-1,:] + array[a/2,:] + array[a/2-2,:] + array[a/2+1,:])/4.0
            else:
                vector = (array[:,b/2-1] + array[:,b/2] + array[:,b/2-2] + array[:,b/2+1])/4.0
            if(semi == True):
                if(i <= 26):
                    index = epoch_class.get_local_extreme(self,vector,max_or_min=False)
                else:
                    index = len(vector)/2
                sub_vector = vector[index:]
                #ilength = len(sub_vector)
                integrate[i] = abs(np.sum(sub_vector)*dx)/(const.D0 * constant)
                #integrate[i] = index
            else:
                integrate[i] = abs(np.sum(vector)*dx/(const.D0 * constant))
        return integrate 
#find local min index
    def get_local_extreme(self,vector,max_or_min = False):
        '''
        This function is used to find a vector's local extreme value.
        parameters:
        vector------input dector
        max_or_min--if find extreme min, set False, else set True
        '''
        import numpy as np
        n = len(vector)
        #find start index
        vector_d = vector
        sub1 = np.argmax(vector_d)
        sub2 = np.argmin(vector_d)
        if(sub1 < sub2):
            index = sub1 + np.argmin(abs(vector[sub1:sub2]))
        else:
            index = sub2 + np.argmin(abs(vector[sub2:sub1]))
        return index
        #while(True):
        #   base = vector[index]
        #    left = vector[index-1]
        #    right = vector[index+1]
        #    if(max_or_min == False):
        #        if((left < base) and (base < right)):
        #            index = index-1
        #        elif(left > base) and (base > right):
        #            index = index+1
        #        else:
        #            return index
        #            break
        #    else:
        #        if((left < base) and (base < right)):
        #            index = index+1
        #        elif(left > base) and (base > right):
        #            index = index-1
        #        else:
        #            return index
        #            break
#find current sheet
    def get_file(self,namelist,field='jz',axes='y',magnitude=False,find_max=True,prefix='1'):
        '''
        This function is used to find max value in which file, and return the index.
        parameters:
        namelist----sdf name list.
        field-------physical field to be integrated, default:'jz'.
        axes--------axes, 'x' or 'y' ,default:'y'.
        magnitude---in integrate a vector's module, set True, default:False
        find_max----if to find max file, set True, default:True
        prefix------file name prefix, default:1
        '''
        import numpy as np
        #use sample sdf to get parameters
        data_dict = epoch_class.get_data(self,namelist[0],prefix=prefix)
        keys = data_dict.keys()
        if(magnitude == False):
            field = epoch_class.get_field(self,field=field,keys=keys)
        else:
            field_d = field
            field = epoch_class.get_field(self,field=field,keys=keys)
        data = np.array((data_dict[field]).data)
        dimen = data.shape
        a = dimen[1]
        b = dimen[0]
        constant = epoch_class.get_constant(self,field=field)
        n = len(namelist)
        info = np.zeros((3,n),np.float)
        #use a loop to open each file and find max value an it's location.
        for i in range(n):
            data_dict = epoch_class.get_data(self,namelist[i],prefix=prefix)
            if(magnitude == False):
                array = epoch_class.get_array(self,data_dict,field=field)/constant
            else:
                array = epoch_class.get_module(self,data_dict,field=field_d)/constant
            if(axes == 'x'):
                vector = (array[a/2-1,:] + array[a/2,:])/2.0
            else:
                vector = (array[:,b/2-1] + array[:,b/2])/2.0
            max_value = np.max(vector)
            index = np.argmax(vector)
            #save into info
            info[0,i] = namelist[i]
            info[1,i] = max_value
            info[2,i] = index
        #find what we want
        if(find_max == True):
            index = np.argmax(info[1,:])    
            return (info[0,index],info[2,index])  
        else:
            return info
#find FWHM
    def get_fwhm(self,filenumber=0,field='Current_Jz',axes='x',magnitude=False,index=256,peak=1.0,\
                 constant=1.0,prefix='1'):
        '''
        This function is used to find FWHM.
        parameters:
        filenumber--sdf file.
        field-------physical field to be integrated, default:'Current_Jz'.
        axes--------axes, 'x' or 'y' ,default:'x'.
        magnitude---in integrate a vector's module, set True, default:False.
        index-------line array index, default=256.
        peak--------max value, default=1.0.
        constant----normalization constant.
        prefix------file name prefix, default:1
        '''
        import numpy as np
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        if(magnitude == True):
            array = epoch_class.get_module(self,data_dict,field=field)/constant
        else:
            array = epoch_class.getdata_array = data_dict[field]
        array = data_array.data

        shape = array.shape
        if(axes == 'x'):
            vector = array[index,:]
            mid = shape[1]/2
        else:
            vector = array[:,index]
            mid = shape[0]
        #then to find FWHM
        vector_abs = np.abs(vector-peak/2.0)
        sub1 = np.argmin(vector_abs)
        vector_abs[sub1] = peak/2.0
        step = 0
        while(True):
            step = step + 1
            sub2 = np.argmin(vector_abs)
            vector_abs[sub2] = peak/2.0
            if(abs(sub2-sub1) > (sub1-mid)):
                break
            else:
                continue
            if(step > 2*mid):
                print 'Find Nothing!'
                break
        if(sub1 < sub2):
            return (sub1,sub2)
        else:
            return (sub2,sub1)
#calculate electron dissipation region.
    def cal_dissipation(self,filenumber,info='current',prefix='1',nspe=5):
        '''
        This function is used to calculate electron dissipation region.
        parameters:
        filenumber--sdf file name.
        info--------information, default:'current'.
        prefix------file name prefix, default:1
        nspe--------n species, default:5.
        '''
        import numpy as np
        from constants import bubble_mr as const
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        #get constant
        j0 = const.J0
        e0 = const.E0
        b0 = const.B0
        v0 = const.V0
        n0 = const.N0
        qe = const.qe
        c = const.c
        #read current
        jx = epoch_class.get_array(self,data_dict,field='Current_Jx')
        jy = epoch_class.get_array(self,data_dict,field='Current_Jy')
        jz = epoch_class.get_array(self,data_dict,field='Current_Jz')
        #read electric field
        ex = epoch_class.get_array(self,data_dict,field='Electric_Field_Ex_averaged')
        ey = epoch_class.get_array(self,data_dict,field='Electric_Field_Ey_averaged')
        ez = epoch_class.get_array(self,data_dict,field='Electric_Field_Ez_averaged')
        #read magnetic field
        bx = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bx_averaged')
        by = epoch_class.get_array(self,data_dict,field='Magnetic_Field_By_averaged')
        bz = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bz_averaged')
        #read electron number density
        rho_pro = []
        rho_ele = []
        for i in range(nspe):
            array_ele = epoch_class.get_array(self,data_dict,field='Derived_Number_Density_ele' + str(i+1))
            array_pro = epoch_class.get_array(self,data_dict,field='Derived_Number_Density_pro' + str(i+1))
            rho_ele = rho_ele + [array_ele]
            rho_pro = rho_pro + [array_pro]
        charge_density = (sum(rho_pro) - sum(rho_ele))*qe
        den_ele = sum(rho_ele)
        if (info == 'current'):
            jxe = []
            jye = []
            jze = []
            for i in range(nspe): 
                array_jx = epoch_class.get_array(self,data_dict,field='Derived_Jx_ele' + str(i+1))
                array_jy = epoch_class.get_array(self,data_dict,field='Derived_Jy_ele' + str(i+1))
                array_jz = epoch_class.get_array(self,data_dict,field='Derived_Jz_ele' + str(i+1))
                jxe = jxe + [array_jx]
                jye = jye + [array_jy]
                jze = jze + [array_jz]
            total_jxe = sum(jxe)
            total_jye = sum(jye)
            total_jze = sum(jze)
            #calculate velocity
            dimen = total_jxe.shape
            p = dimen[0]
            q = dimen[1]
            vx = np.zeros((p,q), np.float)
            vy = np.zeros((p,q), np.float)
            vz = np.zeros((p,q), np.float)
            for i in range(p):
                for j in range(q):
                    if(den_ele[i,j] < 1e24):
                        vx[i,j] = 0
                        vy[i,j] = 0
                        vz[i,j] = 0
                    else:
                        vx[i,j] = -total_jxe[i,j]/den_ele[i,j]/qe
                        vy[i,j] = -total_jye[i,j]/den_ele[i,j]/qe
                        vz[i,j] = -total_jze[i,j]/den_ele[i,j]/qe
        elif(info == 'xz'):
            vx = epoch_class.get_array(self,data_dict,field='Derived_xz_ux_averaged')
            vy = epoch_class.get_array(self,data_dict,field='Derived_xz_uy_averaged')
            vz = epoch_class.get_array(self,data_dict,field='Derived_xz_uz_averaged')
        else:
            pass
        #calculate gamma
        v_module = np.sqrt(vx*vx + vy*vy + vz*vz)/c
        gamma = np.sqrt(1/(1-v_module*v_module))
        #calculate dissipation terms
        #j*e
        je = gamma*(jx*ex + jy*ey + jz*ez)
        #j*v*b
        jvb = gamma*(jx*(vy*bz-vz*by) + jy*(vz*bx-vx*bz) + jz*(vx*by-vy*bx))
        #rho*v*e
        rhove = gamma*(charge_density*(vx*ex + vy*ey + vz*ez))
        #dissipation scaler
        d = je + jvb - rhove
        constant = j0*b0*v0
        return (d/constant,(je+jvb)/constant,je/constant,jvb/constant,-rhove/constant)
#general ohm theory
    def ohm_theory(self,filenumber,axes='x',cut=[512,1024],prefix='1'):
        '''
        This funvtion is used to calculate ohm theory.
        parameters:
        filenumber--sdf file number.
        axes--------axes, 'x' or 'y', default:'x'
        prefix------file name prefix, default:1
        '''
        import numpy as np
        import string
        from constants import bubble_mr as const
        s = 'xy'
        s_index = s.find(axes)
        new_axes = s[1-s_index]
        file_info = epoch_class.get_file(self,namelist=[filenumber],field='jz',axes=new_axes,\
                                       find_max=False)
        index = int(file_info[2])
        a = cut[0]
        b = cut[1]
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        #read data
        #read magnetic field
        ez = epoch_class.get_array(self,data_dict,field='Electric_Field_Ez_averaged')
        bx = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bx')
        by = epoch_class.get_array(self,data_dict,field='Magnetic_Field_By')
        jx = epoch_class.get_array(self,data_dict,field='Current_Jx')
        jy = epoch_class.get_array(self,data_dict,field='Current_Jy')
        vx = epoch_class.get_array(self,data_dict,field='Derived_xz_ux_averaged')
        vy = epoch_class.get_array(self,data_dict,field='Derived_xz_uy_averaged')
        rho = epoch_class.get_array(self,data_dict,field='Derived_Number_Density_ele')
        dpxz = epoch_class.get_array(self,data_dict,field='Derived_xz_DxTxz_averaged_ele')
        dpyz = epoch_class.get_array(self,data_dict,field='Derived_xz_DyTyz_averaged_ele')
        if(axes == 'x'):
            ez = (ez[index-1,a:b] + ez[index,a:b] + ez[index+1,a:b])/3.0
            bx = (bx[index-1,a:b] + bx[index,a:b] + bx[index+1,a:b])/3.0
            by = (by[index-1,a:b] + by[index,a:b] + by[index+1,a:b])/3.0
            jx = (jx[index-1,a:b] + jx[index,a:b] + jx[index+1,a:b])/3.0
            jy = (jy[index-1,a:b] + jy[index,a:b] + jy[index+1,a:b])/3.0
            vx = (vx[index-1,a:b] + vx[index,a:b] + vx[index+1,a:b])/3.0
            vy = (vy[index-1,a:b] + vy[index,a:b] + vy[index+1,a:b])/3.0
            rho = (rho[index-1,a:b] + rho[index,a:b] + rho[index+1,a:b])/3.0
            dpxz = (dpxz[index-1,a:b] + dpxz[index,a:b] + dpxz[index+1,a:b])/3.0
            dpyz = (dpyz[index-1,a:b] + dpyz[index,a:b] + dpyz[index+1,a:b])/3.0
        else:
            pass
        e0 = const.E0
        ez = ez/e0
        vb = -(vx*by - vy*bx)/e0
        pxz = -dpxz/rho/const.qe/e0
        pyz = -dpyz/rho/const.qe/e0
        jb = (jx*by - jy*bx)/rho/const.qe/e0
        array = [ez,vb,pxz,pyz,jb]
        #average if necessary
        n = len(ez)
        new_array = array
        for i in range(3):
            for j in range(5):
                for k in range(n-2):
                    new_array[j][k+1] = (array[j][k] + array[j][k+1] + array[j][k+2])/3.0
        return array
#reconnection rate
    def reconnection_rate(self,namelist,field='Electric_Field_Ez_averaged',axes='y',magnitude=False,\
                          semiwidth=2,prefix='1'):
        '''
        This function is used to calculate reconnection rate.
        parameters:
        namelist----sdf name list.
        field-------physical field to be integrated, default:'Electric_Field_Ez_averaged'.
        axes--------axes, 'x' or 'y' ,default:'y'.
        magnitude---in integrate a vector's module, set True, default:False.
        semiwidth---average semi-width, default:2.
        prefix------file name prefix, default:1
        '''
        import numpy as np
        from constants import bubble_mr as const
        #use sample sdf to get parameters
        data_dict = epoch_class.get_data(self,namelist[0],prefix=prefix)
        keys = data_dict.keys()
        if(magnitude == False):
            field = epoch_class.get_field(self,field=field,keys=keys)
        else:
            field_d = field
            field = epoch_class.get_field(self,field=field,keys=keys)
        data = np.array((data_dict[field]).data)
        dimen = data.shape
        a = dimen[1]
        b = dimen[0]
        constant = epoch_class.get_constant(self,field=field)
        n = len(namelist)
        rate = np.zeros(n,dtype=np.float)
        #determine row or cloumn
        if(axes == 'x'):
            index = a/2
        else:
            index = b/2
        for i in range(n):
            data_dict = epoch_class.get_data(self,namelist[i],prefix=prefix)
            #line_1 = epoch_class.get_line(self,data_dict,field=field,axes=axes,index=index-1)
            #line_2 = epoch_class.get_line(self,data_dict,field=field,axes=axes,index=index)
            #line_3 = epoch_class.get_line(self,data_dict,field=field,axes=axes,index=index-2)
            #line_4 = epoch_class.get_line(self,data_dict,field=field,axes=axes,index=index+1)
            #line = (line_1 + line_2 + line_3 + line_4)/4.0
            ez = epoch_class.get_array(self,data_dict,field=field)
            #calculate convect ez:v*b
            bx = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bx')
            by = epoch_class.get_array(self,data_dict,field='Magnetic_Field_By')
            vx = epoch_class.get_array(self,data_dict,field='Derived_xz_ux_averaged')
            vy = epoch_class.get_array(self,data_dict,field='Derived_xz_uy_averaged')
            ez = ez + (vx*by - vy*bx)
            if(axes == 'y'):
                part_ez = ez[:,index-semiwidth:index+semiwidth-1]
                line_ez = np.sum(part_ez,axis=1)/(2.0*semiwidth)
            else:
                part_ez = ez[index-semiwidth:index+semiwidth-1,:]
                line_ez = np.sum(part_ez,axis=0)/(2.0*semiwidth)
            sub_max = np.argmax(line_ez)
            sub_ez = line_ez[sub_max-semiwidth:sub_max+semiwidth-1]
            rate[i] = np.sum(sub_ez)/(len(sub_ez)*1.0)/constant
        return rate
#general ohm theory
    def dissipation(self,filenumber,prefix='1'):
        '''
        This funvtion is used to calculate ohm theory.
        parameters:
        filenumber--sdf file number.
        prefix------file name prefix, default:1
        '''
        import numpy as np
        import string
        from constants import bubble_mr as const
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        #read data
        #read magnetic field
        ez = epoch_class.get_array(self,data_dict,field='Electric_Field_Ez_averaged')
        bx = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bx')
        by = epoch_class.get_array(self,data_dict,field='Magnetic_Field_By')
        vx = epoch_class.get_array(self,data_dict,field='Derived_xz_ux_averaged')
        vy = epoch_class.get_array(self,data_dict,field='Derived_xz_uy_averaged')
        vb = vx*by - vy*bx
        e0 = const.E0
        array = [ez/e0,vb/e0]
        return array
#u*b
    def cal_ub(self,filenumber,component=3,prefix='1'):
        '''
        This function is used to calculate u*b vector
        parameters:
        filenumber--sdf file number.
        component---x, y, z = 1,2,3 respectively, default:3.
        prefix------file name prefix, default:1
        '''
        import numpy as np
        from constants import bubble_mr as const
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        bx = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bx')
        by = epoch_class.get_array(self,data_dict,field='Magnetic_Field_By')
        bz = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bz')
        vx = epoch_class.get_array(self,data_dict,field='Derived_xz_ux_averaged')
        vy = epoch_class.get_array(self,data_dict,field='Derived_xz_uy_averaged')
        vz = epoch_class.get_array(self,data_dict,field='Derived_xz_uz_averaged')
        ex = epoch_class.get_array(self,data_dict,field='Electric_Field_Ex_averaged')
        ey = epoch_class.get_array(self,data_dict,field='Electric_Field_Ey_averaged')
        ez = epoch_class.get_array(self,data_dict,field='Electric_Field_Ez_averaged')
        ub1 = (vy*bz - vz*by)
        ub2 = (vz*bx - vx*bz)
        ub3 = (vx*by - vy*bx)
        ub = [ex+ub1,ey+ub2,ez+ub3]
        return ub[component-1]/const.E0
#charge density
    def charge_density(self,filenumber,species=3,charge=[1,1,1],prefix='1'):
        '''
        This function is used to calculate charge density.
        parameters:
        filenumber--sdf file number.
        species-----pro species.
        charge------electric charge for each species.
        prefix------file name prefix, default:1
        '''
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        density_ele = epoch_class.get_array(self,data_dict,field='Derived_Number_Density_ele')
        charge_density = -1*density_ele
        for i in range(species):
            field = 'Derived_Number_Density_pro' + str(i+1)
            array = epoch_class.get_array(self,data_dict,field=field)
            charge_density += charge[i]*array
        return charge_density
#calculate electron dissipation region.
    def cal_dissipation_s(self,filenumber,prefix='1'):
        '''
        This function is used to calculate dissipation scaler for different species.
        parameters:
        filenumber--sdf file name.
        prefix------file name prefix, default:1
        '''
        import numpy as np
        from constants import bubble_mr as const
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        #get constant
        j0 = const.J0
        e0 = const.E0
        b0 = const.B0
        v0 = const.V0
        n0 = const.N0
        qe = const.qe
        c = const.c
        #read current
        jx = epoch_class.get_array(self,data_dict,field='Current_Jx')
        jy = epoch_class.get_array(self,data_dict,field='Current_Jy')
        jz = epoch_class.get_array(self,data_dict,field='Current_Jz')
        #read electric field
        ex = epoch_class.get_array(self,data_dict,field='Electric_Field_Ex_averaged')
        ey = epoch_class.get_array(self,data_dict,field='Electric_Field_Ey_averaged')
        ez = epoch_class.get_array(self,data_dict,field='Electric_Field_Ez_averaged')
        #read magnetic field
        bx = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bx')
        by = epoch_class.get_array(self,data_dict,field='Magnetic_Field_By')
        bz = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bz')
        species = ['ele','pro1','pro2','pro3']
        n = len(species)
        #read electron number density
        density_s = []
        rho = 0
        for i in range(n):
            density = epoch_class.get_array(self,data_dict,field='Derived_Number_Density_'+species[i])
            density_s = density_s + [density]
            rho = rho + density_s[i]
        rho = rho - 2*density_s[0]
        #read velocity and calculate D
        D = []
        total_pro = 0
        for i in range(n):
            vx = epoch_class.get_array(self,data_dict,field='Derived_xz_ux_averaged_'+species[i])
            vy = epoch_class.get_array(self,data_dict,field='Derived_xz_uy_averaged_'+species[i])
            vz = epoch_class.get_array(self,data_dict,field='Derived_xz_uz_averaged_'+species[i])
            nie = density_s[i]/density_s[0]
            #calculate gamma
            v_module = np.sqrt(vx*vx + vy*vy + vz*vz)/c
            gamma = np.sqrt(1/(1-v_module*v_module))
            #calculate dissipation terms
            #j*e
            je = nie*gamma*(jx*ex + jy*ey + jz*ez)
            #j*v*b
            jvb = nie*gamma*(jx*(vy*bz-vz*by) + jy*(vz*bx-vx*bz) + jz*(vx*by-vy*bx))
            #rho*v*e
            rhove = nie*gamma*(qe*rho*(vx*ex + vy*ey + vz*ez))
            #dissipation scaler
            d = je + jvb - rhove
            constant = j0*b0*v0
            D = D + [d/constant]
            if(i > 0):
                total_pro = total_pro + d/constant
        D = D + [total_pro]
        return D
#energy spectrum
    def cal_enspe(self,filenumber,species='ele1',info='momentum',ndomain=500,prefix='2',\
                  mass_ratio=1,g_max=1.05,average=False):
        '''
        This function is used to calculate energy spectrum.
        parameters:
        filenumber--sdf file name.
        species-----pro species, default:ele1
        info--------particle information, default:momentum.
        ndomain-----axis step, default:500.
        prefix------file name prefix, default:2
        mass_ratio--mass ratio to electron, default:1
        g_max-------max gamma, default:1.05
        average-----if average, default:False
        '''
        from constants import bubble_mr as const
        import numpy as np
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        #read data
        if(info == 'momentum'):
            m = mass_ratio*const.me
            p0 = m*const.c
            px = epoch_class.get_array(self,data_dict,field='Particles_Px_subset_'+species[:3]+'_'+species)
            py = epoch_class.get_array(self,data_dict,field='Particles_Py_subset_'+species[:3]+'_'+species) 
            pz = epoch_class.get_array(self,data_dict,field='Particles_Pz_subset_'+species[:3]+'_'+species)
            p = np.sqrt(px*px + py*py + pz*pz)
            gamma = np.sqrt(1 + (p/p0)**2)
        if(info == 'gamma'):
            gamma = epoch_class.get_array(self,data_dict,field='Particles_Gamma_subset_'+species[:3]+'_'+species)
        if(info == 'energy'):
            m = mass_ratio*const.me
            e0 = m*const.c*const.c
            energy = epoch_class.get_array(self,data_dict,field='Particles_Energy_subset_'+species[:3]+'_'+species)
            gamma = energy/e0
        #calculate particle number
        weight = epoch_class.get_array(self,data_dict,field='Particles_Weight_subset_'+species[:3]+'_'+species)
        #dg = (max(gamma) - min(gamma))/(ndomain*1.0)
        #g_test = max(gamma)
        #if(g_test < g_max):
        #    g_max = g_test
        #n_step = int((max(gamma) - min(gamma))/ndomain)
        dg = np.linspace(min(gamma),max(gamma),ndomain+1)
        dn = np.zeros(ndomain+1,np.float)
        dgx = dg[1] - dg[0]
        lg = len(gamma)
        ld = len(dg)
        particles = 0
#        for i in range(ndomain):
#            nparticle = 0
#            for j in range(lg):
#                if((gamma[j] > dg[i]) and (gamma[j] <= dg[i+1])):
#                       nparticle += weight[j]
#                       particles += 1
#            dn[i] = nparticle
        for j in range(lg):
            for i in range(ndomain):
                if((gamma[j] > dg[i]) and (gamma[j] <= dg[i+1])):
                    dn[i] += weight[j]
                    break
        #average, if True
        if(average == True):
            for each in range(1,ndomain-1):
                dn[each] = (dn[each-1] + dn[each] + dn[each+1])/3.0
        total_n = sum(dn)
        for each in range(ndomain+1):
            #dn[each] = dn[each]/float(total_n)/dgx
            dn[each] = dn[each]/float(total_n)
            dg[each] = dg[each] - 1
        return (dg,dn,total_n)
#read 3d array
    def get_array_3d(self,data_dict,field='Magnetic_Field_Bx',axis='x',index=0,nspe=3,dimen=2,\
                     axis2='y',index2=0):
        '''
        This function is used to get array frim a sdf dictionary.
        parameters:
        data_dict---data dictionary.
        field-------field, default:'Magnetic_Field_Bx'
        axis--------slice axis, default:'x'
        index-------slice index, default:0
        axis2-------slice axis, default:'y'
        index2------slice index, default:0
        '''
        import numpy as np
        if (field == 'charge_density'):
            ele = []
            pro = []
            for i in range(nspe):
                ele1 = data_dict['Derived_Number_Density_ele' + str(i+1)].data
                pro1 = data_dict['Derived_Number_Density_pro' + str(i+1)].data
                ele = ele + [ele1]
                pro = pro + [pro1]
            #array = pro1 + pro2 - ele1 - ele2
            array = sum(pro) - sum(ele)
        else:
            data_array = data_dict[field]
            array = data_array.data
        if(dimen == 2):
            if(axis == 'x'):
                resu = array[index,:,:]
            elif(axis == 'y'):
                resu = array[:,index,:]
            else:
                resu = array[:,:,index]
        elif(dimen == 1):
            if(axis == 'x'):
                if(axis2 == 'y'):
                    resu = array[index,index2,:]
                else:
                    resu = array[index,:,index2]
            elif(axis == 'y'):
                if(axis2 == 'x'):
                    resu = array[index2,index,:]
                else:
                    resu = array[:,index,index2]
            else:
                if(axis2 == 'x'):
                    resu = array[index2,:,index]
                else:
                    resu = array[:,index2,index]
        else:
            pass
        return np.transpose(resu)
#get 3d extent
    def get_extent_3d(self,data_dict,axis='x'):
        '''
        This function is used to get array extent.
        parameters:
        data_dict---data dictionary read from sdf file.
        axis--------slice axis, default:'x'
        '''
        from constants import proton_radiography as const
        #import numpy as np
        grid = data_dict["Grid_Grid_mid"]
        grid_extent = grid.extents
        xmin = grid_extent[0]/const.di
        xmax = grid_extent[3]/const.di
        ymin = grid_extent[1]/const.di
        ymax = grid_extent[4]/const.di
        zmin = grid_extent[2]/const.di
        zmax = grid_extent[5]/const.di
        if(axis == 'x'):
            extent = [ymin,ymax,zmin,zmax]
        elif(axis == 'y'):
            extent = [xmin,xmax,zmin,zmax]
        else:
            extent = [xmin,xmax,ymin,ymax]
        return extent
#use a dictionary to chose the normalization constant
    def get_constant_3d(self,field='Magnetic_Field_Bx'):
        '''
        This function is used to chose the normalization constant according to the field.
        parameters:
        field-------the field name. default:'Magnetic_Field_Bx'
        '''
        #from constants import laser_mr as const
        from constants import proton_radiography as const
        #from constants import proton_benchmark as const
        normal = {"magnetic":const.B0,"electric":const.E0,\
                  "current":const.J0,"axis":const.di,\
                  "derived_j":const.J0,"density":const.n0,\
                  "temperature":const.T0}
        normal_s = normal.keys()
        for eachone in normal_s:
            if(eachone in field.lower()):
                factor = normal[eachone]
                break
        return factor
#calculate magnetic field tensor force
    def tensor_force(self,filenumber,prefix='1',component=2):
        '''
        This function is used to calculate magnetic field tensor force.
        parameters:
        filenumber--sdf file name.
        prefix------file name prefix, default:1
        component---vector component, default:2(y)
        '''
        from constants import bubble_mr as const
        import numpy as np
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        bx = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bx')
        by = epoch_class.get_array(self,data_dict,field='Magnetic_Field_By')
        bz = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bz')
        b = [bx,by,bz]
        #get dx
        grid = data_dict['Grid_Grid_mid']
        extent = grid.extents
        [m,n] = bx.shape
        dx = (extent[2] - extent[0])/(1.0*n)
        force = np.zeros([m,n])
        #for loop to calculate force
        for i in range(1,m-2):
            for j in range(1,n-2):
                force[i,j] = b[0][i,j]*(b[component-1][i,j+1] - b[component-1][i,j-1])/2.0/dx + \
                             b[1][i,j]*(b[component-1][i+1,j] - b[component-1][i-1,j])/2.0/dx
        force[0,:] = force[1,:]
        force[m-1,:] = force[m-2,:]
        force = force/(const.B0*const.B0/const.di)
        return force
#calculate jb
    def cal_jb(self,filenumber,component=3,prefix='1'):
        '''
        This function is used to calculate u*b vector
        parameters:
        filenumber--sdf file number.
        component---x, y, z = 1,2,3 respectively, default:3.
        prefix------file name prefix, default:1
        '''
        import numpy as np
        from constants import bubble_mr as const
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        bx = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bx')
        by = epoch_class.get_array(self,data_dict,field='Magnetic_Field_By')
        bz = epoch_class.get_array(self,data_dict,field='Magnetic_Field_Bz')
        jx = epoch_class.get_array(self,data_dict,field='Current_Jx')
        jy = epoch_class.get_array(self,data_dict,field='Current_Jy')
        jz = epoch_class.get_array(self,data_dict,field='Current_Jz')
        rho_ele1 = epoch_class.get_array(self,data_dict,field='Derived_Number_Density_ele1')
        rho_ele2 = epoch_class.get_array(self,data_dict,field='Derived_Number_Density_ele2')
        rho_ele3 = epoch_class.get_array(self,data_dict,field='Derived_Number_Density_ele3')
        rho = rho_ele1 + rho_ele2 + rho_ele3
        jb1 = (jy*bz - jz*by)
        jb2 = (jz*bx - jx*bz)
        jb3 = (jx*by - jy*bx)
        jb = [jb1,jb2,jb3]
        return jb[component-1]/rho/const.qe/const.E0
#particle information
    def get_parinfo(self,namelist,field='Px',species='ele',prefix='3',component=1,\
                    id_index=1,mass_ratio=1,time_factor=1.0):
        '''
        This function is used to get particle information.
        parameters:
        namelist----sdf file list.
        field-------physical field, default:Px. #particles_px_subset_tracer_p_tracer_ele.
        species-----species, default:ele
        prefix------file name prefix, default:3
        component---axis, default:1(x).
        id_index----particle id.
        mass_ratio--mass ratio, default:1
        time_factor-time factor, an output sdf file represent time step, default:1.
        '''
        import numpy as np
        from constants import bubble_mr as const
        n = len(namelist)
        array = np.zeros(n,np.float) 
        vz = np.zeros(n,np.float) 
        for i in range(n):
            filenumber = namelist[i]
            data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
            #get ID
            ID = data_dict['Particles_ID_subset_tracer_p_tracer_'+species]
            ID = ID.data
            len_ID = len(ID)
            if((len_ID < id_index) and (i == 0)):
                print 'Too large id_index!'
                break
            else: 
                if((species == 'pro') and (i == 0)):
                    id_index += len_ID
                for j in range(len_ID):
                    if(id_index == ID[j]):
                        index = j
                if(field == 'Grid'):
                    if(component < 4):
                        grid = data_dict['Grid_Particles_subset_tracer_p_tracer_'+species]
                        grid = grid.data
                        #print id_index,'    ',index
                        array[i] = grid[component-1][index]
                    else:
                        pz = data_dict['Particles_Pz_subset_tracer_p_tracer_'+species]
                        g = data_dict['Particles_Gamma_subset_tracer_p_tracer_'+species]
                        pz = pz.data
                        g = g.data
                        vz[i] = pz[index]/g[index]/const.me
                        if(i > 0):
                            array[i] = array[i-1] + ((vz[i] + vz[i-1])/2.0/const.omi*float(time_factor))
                else:
                    data = data_dict['Particles_'+field+'_subset_tracer_p_tracer_'+species]
                    data = data.data
                    #print id_index,'    ',index
                    array[i] = data[index]
        return array
#get scatter variables
    def get_scatter(self,filenumber,field='Pz',species='ele1',prefix='2',component=3,\
                    mass_ratio=1):
        '''
        This function is used to get particle grid and field information.
        parameters:
        filenumber--sdf file number.
        field-------physical field, default:Px. #particles_px_subset_tracer_p_tracer_ele.
        species-----species, default:ele
        prefix------file name prefix, default:3
        component---axis, default:1(x).
        mass_ratio--mass ratio, default:1
        '''
        import numpy as np
        from constants import laser_mr as const
        array = []
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        if(field == 'Grid'):
            grid = data_dict['Grid_Particles_subset_'+species[:3]+'_'+species]
            grid = grid.data
            for j in range(component):
                array = array + [grid[j]/const.la]
            return array
        elif(field == 'weight'):
            weight = data_dict['Particles_Weight_subset_'+species[:3]+'_'+species].data
            return weight
        else:
            fields = ['Px','Py','Pz']
            narray = []
            for i in range(len(fields)):
                data = data_dict['Particles_'+fields[i]+'_subset_'+species[:3]+'_'+species].data
                narray = narray + [data/(const.P0*mass_ratio)]
            return narray
#select data which match condition
    def select_data(self,x=[1],y=[1],z=[1],data=[1],base=[1],least=2.5,weight=[1],cord=True,ifwt=False):
        '''
        This function is used to select data which match the condition.
        parameters:
        x-----------x cordinate.
        y-----------y cordinate.
        z-----------z cordinate.
        data--------data to be seleted.
        base--------base to select.
        least-------least data, default:2.5.
        cord--------if grid, default:True
        '''
        import numpy as np
        n = len(data)
        x_new = []
        y_new = []
        z_new = []
        data_new = []
        weight_new = []
        for i in range(n):
            if(np.abs(base[i]) >= least):
                if(cord == True):
                    x_new.append(x[i])
                    y_new.append(y[i])
                    z_new.append(z[i])
                if(ifwt == True):
                    weight_new.append(weight[i])
                data_new.append(data[i])
        return (np.array(x_new),np.array(y_new),np.array(z_new),np.array(data_new),np.array(weight_new))
#magnetic tension force in 3d
    def tensor_force_3d(self,filenumber,component=3,prefix='2',index=75,axis='y'):
        '''
        This function is used to calculate magnetic field tension force in 3d.
        parameters:
        filenumber--sdf file number.
        component---axis, default:3(means 'z').
        prefix------file name prefix, default:3
        index-------slice index, default:75
        axis--------axis,default:'y'
        '''
        import numpy as np
        from constants import laser_mr as const
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        bx = data_dict['Magnetic_Field_Bx_averaged'].data
        by = data_dict['Magnetic_Field_By_averaged'].data
        bz = data_dict['Magnetic_Field_Bz_averaged'].data
        narray = [bx,by,bz]
        dimen = bx.shape
        #define array about tensor force
        array = np.zeros(dimen,np.float)
        #derive axis step
        grid = data_dict['Grid_Grid_mid'].extents
        dx = (grid[3] - grid[0])/float(dimen[0])
        dy = (grid[4] - grid[1])/float(dimen[1])
        dz = (grid[5] - grid[2])/float(dimen[2])
        #calculate force
        #i = x, j = y, k = z
        xmin = 1
        xmax = dimen[0]-1
        ymin = 1
        ymax = dimen[1]-1
        zmin = 1
        zmax = dimen[2]-1
        
        if(axis == 'x'):
            xmin = index-1
            xmax = index
        elif(axis == 'y'):
            ymin = index-1
            ymax = index
        else:
            zmin = index-1
            zmax = index
        for i in range(xmin,xmax):
            for j in range(ymin,ymax):
                for k in range(zmin,zmax):
                    array[i,j,k] = bx[i,j,k]*(narray[component-1][i+1,j,k] - narray[component-1][i-1,j,k])/(2.0*dx) + by[i,j,k]*(narray[component-1][i,j+1,k] - narray[component-1][i,j-1,k])/(2.0*dy) + bz[i,j,k]*(narray[component-1][i,j,k+1] - narray[component-1][i,j,k-1])/(2.0*dz)
        if(axis == 'x'):
            return array[index-1,:,:]/const.mu0
        elif(axis == 'y'):
            return array[:,index-1,:]/const.mu0
        else:
            return array[:,:,index-1]/const.mu0
#calculate sigma
    def cal_sigma(self,filenumber,species='ele2',component=3,prefix='1',index=75,axis='y'):
        '''
        This function is used to calculate magnetic field tension force in 3d.
        parameters:
        filenumber--sdf file number.
        component---axis, default:3(means 'z').
        prefix------file name prefix, default:3
        index-------slice index, default:75
        axis--------axis,default:'y'
        '''
        import numpy as np
        from constants import laser_mr as const
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        bx = epoch_class.get_array_3d(self,data_dict,field='Magnetic_Field_Bx_averaged',axis=axis,index=index)
        by = epoch_class.get_array_3d(self,data_dict,field='Magnetic_Field_By_averaged',axis=axis,index=index)
        bz = epoch_class.get_array_3d(self,data_dict,field='Magnetic_Field_Bz_averaged',axis=axis,index=index)
        ne = epoch_class.get_array_3d(self,data_dict,field='Derived_Number_Density_'+species,axis=axis,index=index)
        dimen = ne.shape
        sigma = np.zeros(dimen,np.float)
        for i in range(dimen[0]):
            for j in range(dimen[1]):
                if(ne[i,j] > 0.01*const.nc):
                    sigma[i,j] = (bx[i,j]*bx[i,j] + by[i,j]*by[i,j] + bz[i,j]*bz[i,j])/ne[i,j]
        return sigma*const.sigma0
#calculate high energy distribution and average energy
    def cal_en_dis(self,filenumber,field='Pz',species='ele1',prefix='2',component=3,\
                    mass_ratio=1,least=1.632):
        '''
        This function is used to get particle grid and field information.
        parameters:
        filenumber--sdf file number.
        field-------physical field, default:Px. #particles_px_subset_tracer_p_tracer_ele.
        species-----species, default:ele
        prefix------file name prefix, default:3
        component---axis, default:1(x).
        mass_ratio--mass ratio, default:1
        '''
        import numpy as np
        from constants import laser_mr as const
        data = epoch_class.get_scatter(self,filenumber,field=field,species=species,prefix=prefix,component=component,mass_ratio=mass_ratio)
        p = 0
        for i in range(len(data)):
            p = p + data[i]*data[i]
        p = np.sqrt(p)
        weight = epoch_class.get_scatter(self,filenumber,field='weight',species=species,prefix=prefix,component=component,mass_ratio=mass_ratio)
        result = epoch_class.select_data(self,data=p,base=p,weight=weight,cord=False,ifwt=True)
        p_s = result[3]
        w_s = result[4]
        e_s = np.sqrt(1 + p_s*p_s)
        en_ave = np.sum(e_s*w_s)/np.sum(w_s)
        return np.max(p)
#calculate dissipation in 3d
    def cal_diss_je(self,filenumber,species='ele2',prefix='1',component=1,axis='x',index=260):
        '''
        This function is used to calculate j*e dissipation.
        parameters:
        filenumber--sdf file number.
        species-----species, default:ele
        prefix------file name prefix, default:1
        component---axis, default:1(x).
        index-------slice index, default:260
        axis--------axis,default:'x'
        '''
        import numpy as np
        from constants import laser_mr as const
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        current = ['Jx','Jy','Jz']
        electric = ['Ex','Ey','Ez']
        j_f = current[component-1]
        e_f = electric[component-1]
        j = epoch_class.get_array_3d(self,data_dict,field='Current_'+j_f+'_averaged',axis=axis,index=index)
        e = epoch_class.get_array_3d(self,data_dict,field='Electric_Field_'+e_f+'_averaged',axis=axis,index=index)
        dissipation = j*e/const.je0
        return dissipation
#calculate magnetic field energy
    def cal_field_en(self,filenumber,dimen=3,prefix='1',field='magnetic',average=True):
        '''
        This fumction is used to calculate the magnetic field energy.
        parameters:
        filenumber--sdf file number.
        dimen-------dimensions, default:3.
        prefix------file name prefix, default:1
        field-------field, default:'magnetic'.
        average-----if average, default:True
        '''
        import numpy as np
        from constants import laser_mr as const
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        if(dimen == 3):
            if(field == 'magnetic'):
                fields = ['Bx','By','Bz']
            else:
                fields = ['Ex','Ey','Ez']
        else:
            if(field == 'magnetic'):
                fields = ['Bx','By']
            else:
                fields = ['Ex','Ey']
        b2 = 0
        for i in range(len(fields)):
            if(average == True):
                data = data_dict[field[:1].upper()+field[1:]+'_Field_'+fields[i]+'_averaged'].data
                b2 += data**2
            else:
                data = data_dict[field[:1].upper()+field[1:]+'_Field_'+fields[i]].data
                b2 += data**2
        if(field == 'magnetic'):
            b2 = b2/2.0/const.mu0
        else:
            b2 = b2/2.0*const.epsilon0
        en = b2*(const.la/10.0)**3
        return np.sum(en)
#calculate electron's and proton's energy
    def cal_particle_en(self,filenumber,prefix='2',species='ele2',info='gamma',frac=0.5,mass_ratio=1):
        '''
        This fumction is used to calculate the magnetic field energy.
        parameters:
        filenumber--sdf file number.
        species-----species, default:ele
        prefix------file name prefix, default:1
        info--------field information, default:'gamma'
        frac--------output fraction, default:0.5
        mass_ratio--mass ratio, default:1
        '''
        import numpy as np
        from constants import laser_mr as const
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        if(info == 'gamma'):
            gamma = data_dict['Particles_Gamma_subset_'+species[:3]+'_'+species].data
        elif(info == 'p'):
            pass
        else:
            pass
        weight = data_dict['Particles_Weight_subset_'+species[:3]+'_'+species].data
        en = (gamma-1)*mass_ratio*const.en0
        total_en = en*weight/frac
        return np.sum(total_en)
#calculate dissipation term in 3d
    def cal_dissipation_3d(self, filenumber, prefix='1',axis='x', index=[260],nspe=3):
        '''
        This fumction is used to calculate the dissipation term in 3d.
        parameters:
        filenumber--sdf file number.
        prefix------file name prefix, default:1
        index-------slice index, default:260
        axis--------axis,default:'x'
        '''
        import numpy as np
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        nje = []
        nrho = []
        n = len(index)
        for i in range(n):
            #number density
            pro = []
            ele = []
            for j in range(nspe):
                rho_pro1 = epoch_class.get_array_3d(self,data_dict, field='Derived_Number_Density_pro'+str(j+1), axis=axis, index=index[i])
                rho_ele1 = epoch_class.get_array_3d(self,data_dict, field='Derived_Number_Density_ele'+str(j+1), axis=axis, index=index[i])
                pro = pro + [rho_pro1]
                ele = ele + [rho_ele1]
            #current
            jx = epoch_class.get_array_3d(self,data_dict, field='Current_Jx_averaged', axis=axis, index=index[i])
            jy = epoch_class.get_array_3d(self,data_dict, field='Current_Jy_averaged', axis=axis, index=index[i])
            jz = epoch_class.get_array_3d(self,data_dict, field='Current_Jz_averaged', axis=axis, index=index[i])
            jxe = []
            jye = []
            jze = []
            for j in range(nspe):
                jxe1 = epoch_class.get_array_3d(self,data_dict, field='Derived_Jx_ele'+str(j+1), axis=axis, index=index[i])
                jye1 = epoch_class.get_array_3d(self,data_dict, field='Derived_Jy_ele'+str(j+1), axis=axis, index=index[i])
                jze1 = epoch_class.get_array_3d(self,data_dict, field='Derived_Jz_ele'+str(j+1), axis=axis, index=index[i])
                jxe = jxe + [jxe1]
                jye = jye + [jye1]
                jze = jze = [jze1]
            #electric field
            ex = epoch_class.get_array_3d(self,data_dict, field='Electric_Field_Ex_averaged', axis=axis, index=index[i])
            ey = epoch_class.get_array_3d(self,data_dict, field='Electric_Field_Ey_averaged', axis=axis, index=index[i])
            ez = epoch_class.get_array_3d(self,data_dict, field='Electric_Field_Ez_averaged', axis=axis, index=index[i])
            #magnetic field
            bx = epoch_class.get_array_3d(self,data_dict, field='Magnetic_Field_Bx_averaged', axis=axis, index=index[i])
            by = epoch_class.get_array_3d(self,data_dict, field='Magnetic_Field_By_averaged', axis=axis, index=index[i])
            bz = epoch_class.get_array_3d(self,data_dict, field='Magnetic_Field_Bz_averaged', axis=axis, index=index[i])
            #calculate
            pro = sum(pro)
            ele = sum(ele)
            jxe = sum(jxe)
            jye = sum(jye)
            jze = sum(jze)
            #pro = rho_pro2
            #ele = rho_ele2
            #jxe = jxe2
            #jye = jye2
            #jze = jze2
            dimen = pro.shape
            p = dimen[0]
            q = dimen[1]
            vx = np.zeros((p,q), np.float)
            vy = np.zeros((p,q), np.float)
            vz = np.zeros((p,q), np.float)
            gamma = np.zeros((p,q), np.float)
            #calculate velocity and gamma
            qe = 1.602e-19
            c = 3.0e8
            for i in range(p):
                for j in range(q):
                    if(ele[i,j] < 1e26):
                        vx[i,j] = 0
                        vy[i,j] = 0
                        vz[i,j] = 0
                        gamma[i,j] = 1
                    else:
                        vx[i,j] = - jxe[i,j]/ele[i,j]/qe
                        vy[i,j] = - jye[i,j]/ele[i,j]/qe
                        vz[i,j] = - jze[i,j]/ele[i,j]/qe
                        gamma[i,j] = 1.0/np.sqrt(1-(vx[i,j]**2 + vy[i,j]**2 + vz[i,j]**2)/c/c)
            #calculate dissipation
            diss_je = gamma*(jx*(ex+vy*bz-vz*by) + jy*(ey+vz*bx-vx*bz) + jz*(ez+vx*by-vy*bx))
            #diss_je = gamma*(jx*ex + jy*ey + jz*ez)
            diss_rho = - gamma*(pro - ele)*qe*(vx*ex + vy*ey + vz*ez)
            nje = nje + [diss_je]
            nrho = nrho + [diss_rho]
        #average
        diss_je = sum(nje)/float(n)
        diss_rho = sum(nrho)/float(n)
        return (diss_je, diss_rho)
#select label
    def select_label(self, case='laser', axis='x',dimen=3):
        '''
        This function is used to select label.
        case--------aese, default:'laser'.
        axis--------axis, default:'x'.
        '''
        if (case == 'laser'):
            xlabel = '$ X/\lambda_0 $'
            ylabel = '$ Y/\lambda_0 $'
            zlabel = '$ Z/\lambda_0 $'
        else:
            xlabel = '$ X/d_i $'
            ylabel = '$ Y/d_i $'
            zlabel = '$ Z/d_i $'
        #return
        if(axis == 'x'):
            if(dimen == 3):
                return (ylabel, zlabel)
            else:
                return xlabel
        elif(axis == 'y'):
            if(dimen == 3):
                return (xlabel, zlabel)
            else:
                return ylabel
        else:
            if(dimen == 3):
                return (xlabel, ylabel)
            else:
                return zlabel
#divergence field
    def cal_divergence(self, filenumber, field='electeic', prefix='1', axis='x', index=260, ifaverage=True, dimen=3):
        '''
        This fumction is used to calculate the divergence of a field.
        parameters:
        filenumber--sdf file number.
        field-------field, default:'electric'.
        prefix------file name prefix, default:1
        index-------slice index, default:260
        axis--------axis,default:'x'
        ifaverage---if use averaged field, default:True
        '''
        import numpy as np
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        if(ifaverage == True):
            field = ['Ex_averaged', 'Ey_averaged', 'Ez_averaged']
        else:
            field= ['Ex', 'Ey', 'Ez']
        electric = []
        dx = 1.0
        for i in range(dimen):
            array = data_dict['Electric_Field_' + field[i]].data
            electric = electric + [array]
        (a, b, c) = electric[0].shape
        if(axis == 'x'):
            result = np.zeros([b, c], np.float)
            for i in range(1, b-1):
                for j in range(1, c-1):
                    result[i, j] = ((electric[0][index+1,i,j] - electric[0][index-1,i,j]) + (electric[1][index,i+1,j] - electric[1][index,i-1,j]) + (electric[2][index,i,j+1] - electric[2][index,i,j-1]))/2.0/dx
            result[0,:] = result[1,:]
            result[b-1,:] = result[b-2,:]
            result[:,0] = result[:,1]
            result[:,c-1] = result[:, c-2]
        elif(axis == 'y'):
            result = np.zeros([a, c], np.float)
            for i in range(1, a-1):
                for j in range(1, c-1):
                    result[i, j] = ((electric[0][i+1,index,j] - electric[0][i-1,index,j]) + (electric[1][i,index+1,j] - electric[1][i,index-1,j]) + (electric[2][i,index,j+1] - electric[2][i,index,j-1]))/2.0/dx
            result[0,:] = result[1,:]
            result[a-1,:] = result[a-2,:]
            result[:,0] = result[:,1]
            result[:,c-1] = result[:, c-2]
        else:
            result = np.zeros([a, b], np.float)
            for i in range(1, a-1):
                for j in range(1, b-1):
                    result[i, j] = ((electric[0][i+1,j,index] - electric[0][i-1,j,index]) + (electric[1][i,j+1,index] - electric[1][i,j-1,index]) + (electric[2][i,j,index+1] - electric[2][i,j,index-1]))/2.0/dx
            result[0,:] = result[1,:]
            result[a-1,:] = result[a-2,:]
            result[:,0] = result[:,1]
            result[:,b-1] = result[:, b-2]
        return np.transpose(result)
#vortex
    def cal_vortex(self, filenumber, field='electeic', prefix='1', axis='x', index=260, ifaverage=True, dimen=3, component=1):
        '''
        This fumction is used to calculate the divergence of a field.
        parameters:
        filenumber--sdf file number.
        field-------field, default:'electric'.
        prefix------file name prefix, default:1
        index-------slice index, default:260
        axis--------axis,default:'x'
        ifaverage---if use averaged field, default:True
        '''
        import numpy as np
        data_dict = epoch_class.get_data(self,filenumber,prefix=prefix)
        if(ifaverage == True):
            field = ['Ex_averaged', 'Ey_averaged', 'Ez_averaged']
        else:
            field= ['Ex', 'Ey', 'Ez']
        electric = []
        dx = 1.0
        for i in range(dimen):
            array = data_dict['Electric_Field_' + field[i]].data
            electric = electric + [array]
        (a, b, c) = electric[0].shape
        if(axis == 'x'):
            result = np.zeros([b, c], np.float)
            for i in range(1, b-1):
                for j in range(1, c-1):
                    if(component == 1):
                        result[i, j] = ((electric[2][index,i+1,j] - electric[2][index,i-1,j]) - (electric[1][index,i,j+1] - electric[1][index,i,j-1]))/2.0/dx
                    elif(component == 2):
                        result[i, j] = ((electric[0][index,i,j+1] - electric[0][index,i,j-1]) - (electric[2][index+1,i,j] - electric[2][index-1,i,j]))/2.0/dx
                    else:
                        result[i, j] = ((electric[1][index+1,i,j] - electric[1][index-1,i,j]) - (electric[0][index,i+1,j] - electric[0][index,i-1,j]))/2.0/dx
            result[0,:] = result[1,:]
            result[b-1,:] = result[b-2,:]
            result[:,0] = result[:,1]
            result[:,c-1] = result[:, c-2]
        elif(axis == 'y'):
            result = np.zeros([a, c], np.float)
            for i in range(1, a-1):
                for j in range(1, c-1):
                    if(component == 1):
                        result[i, j] = ((electric[2][i,index+1,j] - electric[2][i,index-1,j]) - (electric[1][i,index,j+1] - electric[1][i,index,j-1]))/2.0/dx
                    elif(component == 2):
                        result[i, j] = ((electric[0][i,index,j+1] - electric[0][i,index,j-1]) - (electric[2][i+1,index,j] - electric[2][i-1,index,j]))/2.0/dx
                    else:
                        result[i, j] = ((electric[1][i+1,index,j] - electric[1][i-1,index,j]) - (electric[0][i,index+1,j] - electric[0][i,index-1,j]))/2.0/dx
            result[0,:] = result[1,:]
            result[a-1,:] = result[a-2,:]
            result[:,0] = result[:,1]
            result[:,c-1] = result[:, c-2]
        else:
            result = np.zeros([a, b], np.float)
            for i in range(1, a-1):
                for j in range(1, b-1):
                    if(component == 1):
                        result[i, j] = ((electric[2][i,j+1,index] - electric[2][i,j-1,index]) - (electric[1][i,j,index+1] - electric[1][i,j,index-1]))/2.0/dx
                    elif(component == 2):
                        result[i, j] = ((electric[0][i,j,index+1] - electric[0][i,j,index-1]) - (electric[2][i+1,j,index] - electric[2][i-1,j,index]))/2.0/dx
                    else:
                        result[i, j] = ((electric[1][i+1,j,index] - electric[1][i-1,j,index]) - (electric[0][i,j+1,index] - electric[0][i,j-1,index]))/2.0/dx
            result[0,:] = result[1,:]
            result[a-1,:] = result[a-2,:]
            result[:,0] = result[:,1]
            result[:,b-1] = result[:, b-2]
        return np.transpose(result)


# -*- coding: utf-8 -*-
#*************************************************************************
#***File Name: epoch_visual.py
#***Author: Zhonghai Zhao
#***Mail: zhaozhonghi@126.com 
#***Created Time: 2018年03月25日 星期日 14时39分05秒
#*************************************************************************
class epoch_visual(object):
    '''
    This class contains some functions to visualize sdf file.
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
    #default_label
    def default_label(self):
        '''
        This function stored some default labels.
        '''
        label=['mass-ratio=','100:100','100:25','100:400']
        return label
    def plot_array(self,filenumber=0,info='dissipation_3d',field='electric',display=True,factor=1.0,axis='x',index=260, prefix='1',vlim=0.5,arti_v=False, case=0,nspe=3,ifaverage=True,dimen=3,component=3):
        '''
        This function is used to visualize 2d field data.
        parameters:
        filenumber--sdf file number, an integer, default:0.
        info--------array information.
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        factor------vmax multifactor,less than 1, default:1.0
        axis--------slice axis, default:'x'
        index-------slice index, default:0
        prefix------file name prefix, default:1
        vlim--------limit range, default:0.5
        arti_v------limit range artificial, default:False.
        case--------different case
        nspe--------n species, default:3.
        '''
        import numpy as np
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from epoch_class import epoch_class as sdfclass
        sc = sdfclass()
        #get name list
        namelist = sc.get_list(filenumber)
        n = len(namelist)
        dimen = info[-2:]
        #get sample parameters
        data_dict = sc.get_data(namelist[0],prefix=prefix)
        #get extent
        if (dimen == '3d'):
            extent = sc.get_extent_3d(data_dict,axis=axis)
        elif (dimen == '2d'):
            extent = sc.get_extent(data_dict)
        else:
            pass
        #get array
        narray = []
        title = []
        path = []
        #select case
        if(info == 'dissipation_3d'):
            from constants import laser_mr as const
            coeff = const.J0 * const.E0
            for i in range(n):
                array = sc.cal_dissipation_3d(namelist[i], prefix='1', axis=axis, index=index,nspe=nspe)
                if(case == 0):
                    array = array[0] + array[1]
                    title = title +  ['Dissipation ' + '$ D_e $' + ' time = ' + str(namelist[i]+29) + '$ T_0 $']
                elif(case == 1):
                    array = array[0]
                    title = title +  [r'$ \vec J\cdot (\vec E + \vec v_e \times \vec B) $' + ' time = ' + str(namelist[i]+29) + '$ T_0 $']
                elif(case == 2):
                    array = array[1]
                    title = title +  [r'$ - \rho_c \cdot (\vec v_e \cdot \vec E) $' + ' time = ' + str(namelist[i]+29) + '$ T_0 $']
                else:
                    pass
                narray = narray + [array/coeff]
                path = path + ['dissipation/Dissipation_' + str(namelist[i]+29) + '_' + axis + '_' + str(index) + '.png']
            #others
            labels = sc.select_label(case='laser', axis=axis)
            xlabel = labels[0]
            ylabel = labels[1]
        elif(info == 'divergence_3d'):
            from constants import laser_mr as const
            coeff = (const.qe*const.nc)/const.epsilon0*(const.la/10.0)
            for i in range(n):
                array = sc.cal_divergence(namelist[i], field=field, prefix=prefix, axis=axis, index=index, ifaverage=ifaverage, dimen=3)
                narray = narray + [array/coeff]
                title = title + [r'$ \nabla \cdot \vec E $' + ' time = ' + str(namelist[i]) + '$ T_0 $']
                path = path + ['divergence_' + field + '_' + str(namelist[i]) + '_' + axis + '_' + str(index) + '.png']
            labels = sc.select_label(case='laser', axis=axis)
            xlabel = labels[0]
            ylabel = labels[1]
        elif(info == 'vortex_3d'):
            from constants import laser_mr as const
            coeff = (const.qe*const.nc)/const.epsilon0*(const.la/10.0)
            for i in range(n):
                array = sc.cal_vortex(namelist[i], field=field, prefix=prefix, axis=axis, index=index, ifaverage=ifaverage, dimen=3, component=component)
                narray = narray + [array/coeff]
                if(component == 1):
                    title = title + [r'$ (\nabla \times \vec E)_x $' + ' time = ' + str(namelist[i]) + '$ T_0 $']
                elif(component == 2):
                    title = title + [r'$ (\nabla \times \vec E)_y $' + ' time = ' + str(namelist[i]) + '$ T_0 $']
                else:
                    title = title + [r'$ (\nabla \times \vec E)_z $' + ' time = ' + str(namelist[i]) + '$ T_0 $']
                path = path + ['vortex_' + field + '_' + str(namelist[i]) + '_' + axis + '_' + str(index) + '.png']
            labels = sc.select_label(case='laser', axis=axis)
            xlabel = labels[0]
            ylabel = labels[1]
        else:
            pass
        #find max
        if (arti_v == False):
            total = np.abs(np.array(narray))
            vmax = total.max()*factor
        else:
            vmax = vlim
        #select cmp and vmin
        sample = np.array(narray[0])
        if(np.min(sample) < 0):
            cmap = cm.RdBu_r
            vmin = -vmax
        else:
            cmap = cm.Blues
            vmin = 0
        #plot all the figure
        plt.figure(figsize=(10,5))
        for i in range(n):
            ax = plt.gca()
            #im = ax.contourf(array,100,extent=extent,origin='lower',cmap=cmap)
            im = ax.imshow(narray[i],extent=extent,origin='lower',cmap=cmap,vmax=vmax,vmin=vmin,\
                           interpolation='spline36')
            #add label
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            #add title
            plt.title(title[i])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right',size='3%',pad=0.1)
            plt.colorbar(im,cax=cax)
            #select display or save picture
            if(display == True):
                plt.show()
            else:
                s1 = "figure/"
                plt.savefig(s1+path[i],dpi=300)
                if (i == n-1):
                    plt.close()
                else:
                    plt.clf()
        #return narray
#line plot
    def plot_line(self,filenumber=0,info='dissipation_3d',field='bz',display=True,factor=1.0,axis='x',index=260,axis2='y',index2=0, prefix='1',lim=[-1,1],arti_v=False,nspe=3, figure_size=(10, 5)):
        '''
        This function is used to plot line.
        parameters:
        filenumber--sdf file number, an integer, default:0.
        info--------array information.
        field-------physical field, default:bz.
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        factor------vmax multifactor,less than 1, default:1.0
        axis--------slice axis, default:'x'
        index-------slice index, default:260
        axis2-------slice axis, default:'y'
        index2------slice index, default:0
        prefix------file name prefix, default:1
        lim---------limit range, default:[-1,1]
        arti_v------limit range artificial, default:False.
        nspe--------n species, default:3.
        '''
        import numpy as np
        from matplotlib import pyplot as plt
        from epoch_class import epoch_class as sdfclass
        sc = sdfclass()
        color_set = sc.line_set()
        #get name list
        namelist = sc.get_list(filenumber)
        n = len(namelist)
        dimen = info[-2:]
        #get sample parameters
        data_dict = sc.get_data(namelist[0],prefix=prefix)
        keys = data_dict.keys()
        #get x_axis
        if (dimen == '3d'):
            cord = 'xyz'
            for each in cord:
                if((each != axis) and (each != axis2)):
                    axes = each
            x_axis = sc.get_axis_3d(data_dict,axes=axes)
        elif (dimen == '2d'):
            cord = 'xy'
            for each in cord:
                if(each != axis):
                    axes = each
            x_axis = sc.get_axis(data_dict,axes=axes)
        elif (dimen == '1d'):
            axes = axis
            x_axis = sc.get_axis(data_dict, dimen=dimen)
        else:
            pass
        #get array
        narray = []
        legend = []
        #select case
        if(info == 'dissipation_3d'):
            pass
        else:
            if(dimen == '3d'):
                from constants import laser_mr as const
                field = sc.get_field(field=field,keys=keys)
                constant = sc.get_constant_3d(field=field)
                x0 = const.la
                x_axis = x_axis/x0
                # get array
                for i in range(n):
                    data_dict = sc.get_data(namelist[i],prefix=prefix)
                    array = sc.get_array_3d(data_dict,field=field,axis=axis,index=index,axis2=axis2,index2=index2,dimen=1)
                    narray = narray + [array/constant]
                    legend = legend + ['t = ' + str(namelist[i]) + '$ T_0 $']
                #others
                label = sc.select_label(case='laser', axis=axes, dimen=1)
                title = field
            elif(dimen == '2d'):
                pass
            elif(dimen == '1d'):
                from constants import bubble_expansion as const
                field = sc.get_field(field=field,keys=keys)
                constant = sc.get_constant_3d(field=field)
                x0 = const.d0
                x_axis = x_axis/x0
                # get array
                for i in range(n):
                    data_dict = sc.get_data(namelist[i],prefix=prefix)
                    array = data_dict[field].data
                    narray.append(array/constant)
                    legend = legend + ['t = ' + str(namelist[i]) + '$ T_0 $']
                #others
                label = sc.select_label(case='others', axis=axes, dimen=1)
                title = field
            else:
                pass
        #find max
        if (arti_v == False):
            total = np.array(narray)
            vmax = total.max()*factor
            vmin = total.min()*factor
        else:
            vmax = lim[1]
            vmin = lim[0]
        #plot all the figure
        plt.figure(figsize=figure_size)
        for i in range(n):
            plt.plot(x_axis,narray[i],color_set[i],label=legend[i])
        #add label
        plt.xlabel(label)
        #add others
        plt.axis([x_axis.min(), x_axis.max(), vmin, vmax])
        plt.title(title)
        plt.legend()
        #select display or save picture
        if(display == True):
            plt.show()
        else:
            s1 = "figure/"
            s2 = field + '.png' 
            plt.savefig(s1+s2,dpi=300)
            plt.clf()
        plt.close()

#plot 2d field
    def implot(self,filenumber=0,field='bx',magnitude=False,display=True,factor=1.0,gf=False,\
               g_field=150,average=False,prefix='1'):
        '''
        This function is used to visualize 2d field data.
        parameters:
        filenumber--sdf file number, an integer, default:0.
        field-------physical field, default:'bx'.
        magnitude---if plot a vector's module, set True, default:False.
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        factor------vmax multifactor,less than 1, default:1.0
        gf----------guide field, default:False
        g_field-----guide field, default:150
        prefix------file name prefix, default:1
        '''
        import numpy as np
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from epoch_class import epoch_class as sdfclass
        sc = sdfclass()
        #get name list
        namelist = sc.get_list(filenumber)
        n = len(namelist)
        #get sample parameters
        data_dict = sc.get_data(namelist[0],prefix=prefix)
        keys = data_dict.keys()
        if(magnitude == False):
            field = sc.get_field(field=field,keys=keys)
        constant = sc.get_constant(field=field)
        #dx = sc.get_constant(field='axis')
        extent = sc.get_extent(data_dict)
        #for loop to get array from every sdf file
        if(average == False):
            k = 0
        else:
            k = 1
        narray = []
        for eachone in namelist:
            t_array = 0
            for j in range(eachone-k,eachone+k+1):
                data_dict = sc.get_data(j,prefix=prefix)
                if(magnitude == False):
                    array = sc.get_array(data_dict,field=field,gf=gf,g_field=g_field)/constant
                else:
                    array = sc.get_module(data_dict,field=field,gf=gf,g_field=g_field)/constant
                t_array = t_array + array
            narray = narray + [t_array/(2.0*k+1.0)]
        #find max
        total = np.abs(np.array(narray))
        vmax = total.max()*factor
        #select cmp and vmin
        sample = np.array(narray[0])
        if(np.min(sample) < 0):
            cmap = cm.RdBu_r
            vmin = -vmax
        else:
            cmap = cm.Blues
            vmin = 0
        #plot all the figure
        plt.figure(figsize=(10,5))
        for i in range(n):
            #plt.figure(figsize=(10,5))
            ax = plt.gca()
            #im = ax.contourf(array,100,extent=extent,origin='lower',cmap=cmap)
            im = ax.imshow(narray[i],extent=extent,origin='lower',cmap=cmap,vmax=vmax,vmin=vmin,\
                           interpolation='spline36')
            #add label
            plt.xlabel("$ X/d_i $")
            plt.ylabel("$ Y/d_i $")
            #add title
            t1 = "$ time = "
            t2 = str(namelist[i])
            t3 = "\omega_{ci}^{-1} $"
            t = field+"  "+t1+t2+"  "+t3
            plt.title(t)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right',size='3%',pad=0.1)
            plt.colorbar(im,cax=cax)
            #select display or save picture
            if(display == True):
                plt.show()
            else:
                s1 = "figure/"
                snumber = str(namelist[i])
                s2 = snumber.zfill(4)
                s3 = ".png"
                path = s1+field+s2+s3
                plt.savefig(path,dpi=300)
                #plt.close()
                plt.clf()
        #return narray
    #plot line array
    def line_plot(self,filenumber=0,field='bx',magnitude=False,display=True,axes='x',pathnumber=1,\
                  prefix='1'):
        '''
        This function is used to plot line array.
        parameters:
        filenumber---filenumber--sdf file number, an integer, default:0.
        field-------physical field, default:'bx'.
        magnitude---if plot a vector's module, set True, default:False.
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        axes--------axes,'x' or 'y', default:'x'. 
        pathnumber--save file, default:1
        prefix------file name prefix, default:1
        '''
        import numpy as np
        from matplotlib import pyplot as plt
        from epoch_class import epoch_class as sdfclass
        sc = sdfclass()
        #get name list
        namelist = sc.get_list(filenumber)
        n = len(namelist)
        #get sample parameters
        data_dict = sc.get_data(namelist[0],prefix=prefix)
        keys = data_dict.keys()
        if(magnitude == False):
            field = sc.get_field(field=field,keys=keys)
            index = sc.get_index(data_dict,field=field,axes=axes)
        else:
            index = sc.get_index(data_dict,field='Derived_Number_Density_ele',axes=axes)
        constant = sc.get_constant(field=field)
        axis = sc.get_axis(data_dict,axes=axes)
        #for loop to get array from every sdf file
        nvector = []
        for eachone in namelist:
            data_dict = sc.get_data(eachone,prefix=prefix)
            if(magnitude == False):
                vector = sc.get_line(data_dict,field=field,axes=axes,index=index)/constant
            else:
                array = sc.get_module(data_dict,field=field)/constant
                vector = sc.get_module_line(array,axes=axes,index=index)
            nvector = nvector + [vector]
        #find max
        total = np.abs(np.array(nvector))
        vmax = total.max()
        #select cmp and vmin
        sample = np.array(nvector[0])
        if(np.min(sample) <= 0):
            vmin = -vmax
        else:
            vmin = 0
        color_set = sc.line_set()
        plt.figure(figsize=(10,5))
        for i in range(n):
            plt.plot(axis,nvector[i],color_set[i],label='t='+str(namelist[i]))
        plt.xlabel('$ '+axes.upper()+'/d_i $')
        plt.ylabel(field.upper())
        plt.axis([axis.min(),axis.max(),vmin,vmax])
        plt.legend()
        if(display == True):
            plt.show()
        else:
            s1 = "figure/"
            s2 = field+str(pathnumber)
            s3 = ".png"
            path = s1+s2+s3
            plt.savefig(path,dpi=300)
            plt.clf()
        #return nvector
    #plot reconnection flux
    def integrate_plot(self,namelist,field='bx',axes='y',magnitude=False,semi=True,\
                       label=['mass-ratio=','100:100','100:25','100:400','25:400'],display=True,\
                       max_or_min=False,folder=[4,1]):
        '''
        This function is use to plot integrate flux.
        parameters:
        namelist----sdf name list.
        field-------physical field to be integrated, default:'Magnetic_Field_Bx'
        axes--------axes, 'x' or 'y' ,default:'y'.
        magnitude---in integrate a vector's module, set True, default:False
        semi--------if integrate semi-axis, set True, default:True.
        label-------default label.
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        max_or_min--when semi is True, if find extreme min, set False, else set True.
        folder------file path, [folder number,folder start], default:[4,1]
        '''
        import os
        from epoch_class import epoch_class as mysdf
        from matplotlib import pyplot as plt
        sc = mysdf()
        col_sty = sc.line_set()
        cwd = os.getcwd()
        s = '/data'
        plt.figure(figsize=(10,5))
        for i in range(folder[0]):
            #set path
            path = cwd + s + str(i+folder[1])
            os.chdir(path)
            vector = sc.line_integrate(namelist,field=field,axes=axes,magnitude=magnitude,semi=semi,\
                                       max_or_min=max_or_min)
            plt.plot(vector,col_sty[i],label=label[0]+label[i+1])
        plt.xlabel('$ t/\omega_{ci}^{-1} $')
        plt.ylabel('$ \psi/B_0d_i $')
        plt.title('$ Reconnection $'+'  '+'$ Flux $')
        plt.legend()
        #change to the old directory
        os.chdir(cwd)
        if(display == True):
            plt.show()
        else:
            s1 = 'figure/'
            s2 = 'FLUX_'
            s3 = axes.upper()
            s4 = '.png'
            path = s1+s2+s3+s4
            plt.savefig(path,dpi=300)
            plt.clf()
    #plot differrent case in a single figure
    def plot_s(self,namelist,field='jz',axes='y',magnitude=False,display=True,find_max=True,\
               label=['mass-ratio=','100:100','100:25','100:400','25:400'],folder=[4,1],\
               prefix='1'):
        '''
        This function is used to plot lines in a single figure.
        parameters:
        namelist----sdf name list.
        field-------physical field to be integrated, default:'jz'.
        axes--------axes, 'x' or 'y' ,default:'y'.
        magnitude---in integrate a vector's module, set True, default:False.
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        find_max----if to find max file, set True, default:True
        prefix------file name prefix, default:1
        '''
        import os
        from epoch_class import epoch_class as mysdf
        from matplotlib import pyplot as plt
        import string
        import numpy as np
        sc = mysdf()
        col_sty = sc.line_set()
        cwd = os.getcwd()
        s = '/data'
        #
        plt.figure(figsize=(10,5))
        for i in range(folder[0]):
            #set path
            path = cwd + s + str(i+folder[1])
            os.chdir(path)
            #use a sample sdf file to get axis
            data_dict = sc.get_data(namelist[0],prefix=prefix)
            info = sc.get_file(namelist,field=field,axes=axes,magnitude=magnitude,find_max=True)
            #get array
            number = int(info[0])
            index = int(info[1])
            data_dict = sc.get_data(number,prefix=prefix)
            keys = data_dict.keys()
            if(magnitude == False):
                field = sc.get_field(field=field,keys=keys)
                array = np.transpose((data_dict[field]).data)
            else:
                field_d = field
                array = sc.get_module(data_dict,field=field_d)
            constant = sc.get_constant(field=field)
            #get vector
            if(axes == 'x'):
                vector = array[:,index]/constant
            else:
                vector = array[index,:]/constant
            #get axis
            cord = 'xy'
            sub = cord.find(axes)
            new_axes = cord[1-sub]
            axis = sc.get_axis(data_dict,axes=new_axes)
            plt.plot(axis,vector,col_sty[i],label=label[0]+label[i+1])
        os.chdir(cwd)
        plt.xlabel('$ '+new_axes.upper()+'/d_i $')
        plt.ylabel(field.upper())
        plt.legend()
        if(display == True):
            plt.show()
        else:
            s1 = 'figure/'
            s2 = field.upper()
            s3 = new_axes.upper()
            s4 = '.png'
            path = s1+s2+'_'+s3+s4
            plt.savefig(path,dpi=300)
            plt.clf()
    #plot dissipation scaler
    def plot_dissipation(self,namelist,display=True,info='current',factor=0.5,prefix='1',nspe=5):
        '''
        This function is used to plot dissipation scalar.
        parameters:
        namelist----sdf name list.
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        info--------information, default:'current'.
        prefix------file name prefix, default:1
        nspe--------n species, default:5.
        '''
        from epoch_class import epoch_class as mysdf
        import numpy as np
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        sc = mysdf()
        namelist = sc.get_list(namelist)
        n = len(namelist)
        #species = ['ele','pro1','pro2','pro3']
        #use sample data to get extent
        data_dict = sc.get_data(namelist[0],prefix=prefix)
        extent = sc.get_extent(data_dict)
        title = [r'$ Dissipation $'+'  '+'$ Scalar $',r'$ \vec J\cdot (\vec E + \vec v \times \vec B) $',r'$ \vec J\cdot \vec E $',r'$ \vec J\cdot (\vec v \times \vec B) $',r'$ \rho_c(\vec v \cdot \vec E) $']
        for i in range(n):
            vector = sc.cal_dissipation(namelist[i],info=info,prefix=prefix,nspe=nspe)
            vmax = np.max(np.array(vector))*factor
            if(np.min(np.array(vector)) <= 0):
                cmap = cm.RdBu_r
                vmin = -vmax
            else:
                cmap = cm.Blues
                vmin = 0
            for j in range(len(vector)):
                plt.figure(i,figsize=(10,5))
                ax = plt.gca()
                #im = ax.contourf(array,100,extent=extent,origin='lower',cmap=cmap)
                im = ax.imshow(vector[j],extent=extent,origin='lower',cmap=cmap,vmax=vmax,vmin=vmin,\
                               interpolation='spline36')
                #add label
                plt.xlabel("$ X/d_i $")
                plt.ylabel("$ Y/d_i $")
                #add title
                t1 = "$ time = "
                t2 = str(namelist[i])
                t3 = "\omega_{ci}^{-1} $"
                t = title[j]+"  "+t1+t2+"  "+t3
                plt.title(t)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right',size='3%',pad=0.1)
                plt.colorbar(im,cax=cax)
                #display or save figure
                if(display == True):
                    plt.show()
                else:
                    s1 = 'figure/dissipation/'
                    s2 = "Dissipation_"
                    s3 = str(namelist[i])+'_'+str(j+1)
                    s4 = '.png'
                    path = s1+s2+s3+s4
                    plt.savefig(path,dpi=300)
                    plt.clf()
    #plot dissipation line
    def plot_dissipation_line(self,namelist,axes='x',display=True,factor=0.5,index=256,\
                              prefix='1'):
        '''
        This function is used to plot dissipation line and calculate percent.
        namelist----sdf name list.
        axes--------axes, 'x' or 'y', default:'x'
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        factor------limit factor, default:0.5
        prefix------file name prefix, default:1
        '''
        from epoch_class import epoch_class as mysdf
        import numpy as np
        from matplotlib import pyplot as plt
        sc = mysdf()
        namelist = sc.get_list(namelist)
        n = len(namelist)
        percent = np.zeros((4,n),np.float)
        #use sample data to get axis
        data_dict = sc.get_data(namelist[0],prefix=prefix)
        axis = sc.get_axis(data_dict,axes=axes)
        label = [r'$ Dissipation $'+'  '+'$ Scalar $',r'$ \vec J\cdot \vec E $',r'$ \vec J\cdot (\vec v_e \times \vec B) $',r'$ \rho_c(\vec v_e \cdot \vec E) $']
        col_sty = sdf_visual.line_set(self)
        for i in range(n):
            vector = sc.cal_dissipation(namelist[i])
            #get shape
            shape = (vector[0]).shape
            a = shape[0]
            b = shape[1]
            #get line
            if(axes == 'x'):
                array = np.zeros((4,b),np.float)
                for j in range(4):
                    array[j] = (vector[j][index,:] + vector[j][index,:])/2.0
                    #average if necessary
                    for k in range(b-2):
                        array[j,k+1] = (array[j,k] + array[j,k+1] + array[j,k+2])/3.0
            else:
                array = np.zeros((4,a),np.float)
                for j in range(4):
                    array[j] = (vector[j][:,b/2-1] + vector[j][:,b/2])/2.0
                    #average if necessary
                    for k in range(a-2):
                        array[j,k+1] = (array[j,k] + array[j,k+1] + array[j,k+2])/3.0
            #calculate percent
            for j in range(4):
                percent[j,i] = np.sum(array[j])/np.sum(array[0])
            #plor figure
            plt.figure(i,figsize=(10,5))
            for k in range(4):
                plt.plot(axis,array[k],col_sty[k],label=label[k])
            plt.xlabel('$ '+axes.upper()+'/d_i $')
            plt.ylabel("Dissipation Scalar")
            #plt.axis([axis.min(),axis.max(),vmin,vmax])
            plt.legend()
            if(display == True):
                plt.show()
            else:
                s1 = "figure/"
                s2 = "Dissipation_"+axes.upper()+str(namelist[i])
                s3 = ".png"
                path = s1+s2+s3
                plt.savefig(path,dpi=300)
                plt.clf()
            return percent
#general ohm theory
    def plot_ohm(self,namelist,axes='x',domain='q',display=True,prefix='1'):
        '''
        This funvtion is used to calculate ohm theory.
        parameters:
        namelist----sdf name list.
        axes--------axes, 'x' or 'y', default:'x'
        domain------range domain,'w':whole, 'h':half, 'q':quater, default:'q'
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        prefix------file name prefix, default:1
        '''
        import numpy as np
        from bubble_mr import bubble_mr as const
        from matplotlib import pyplot as plt
        from epoch_class import epoch_class as mysdf
        sc = mysdf()
        col_sty = sdf_visual.line_set(self)
        label = [r'$ E_z $',r'$ (\vec v\times \vec B)_z $',r'$ D_xP_{exz} $',r'$ D_yP_{eyz} $',\
                 r'$ (\vec J\times \vec B)_z $']
        info = {'w':2,'h':4,'q':8}
        info_keys = info.keys()
        #use sample data to get base imformation
        namelist = sc.get_list(namelist)
        data_dict = sc.get_data(namelist[0],prefix=prefix)
        ez =  sc.get_array(data_dict,field='Electric_Field_Ez')
        shape = ez.shape
        if(axes == 'x'):
            length = shape[1]
        else:
            length = shape[0]
        m = info[domain]
        a = length/2 - length/m
        b = length/2 + length/m -1
        axis = sc.get_axis(data_dict,axes=axes)
        #get actual axis
        axis = axis[a:b]
        n = len(namelist)
        for i in range(n):
            array = sc.ohm_theory(namelist[i],axes=axes,cut=[a,b])
            plt.figure(figsize=(10,5))
            for j in range(len(array)):
                plt.plot(axis,array[j],col_sty[j],label=label[j])
                plt.xlabel('$ '+axes.upper()+'/d_i $')
                plt.ylabel("$ E_z $")
                t1 = "$ time = "
                t2 = str(namelist[i])
                t3 = "\omega_{ci}^{-1} $"
                t = t1+t2+"  "+t3
                plt.title(t)
                plt.legend()
            if(display == True):
                plt.show()
            else:
                s1 = "figure/"
                s2 = "Ohm_"+str(namelist[i])
                s3 = ".png"
                path = s1+s2+s3
                plt.savefig(path,dpi=300)
                plt.clf()
       # return array
#reconnection rate
    def plot_reconnection_rate(self,namelist,field='Electric_Field_Ez_averaged',axes='y',\
                               label=['mass-ratio=','100:100','100:25','100:400','25:400'],\
                               display=True,magnitude=False,folder=[4,1],semiwidth=2):
        '''
        This function is used to calculate reconnection rate.
        parameters:
        namelist----sdf name list.
        field-------physical field to be integrated, default:'Electric_Field_Ez_averaged'.
        axes--------axes, 'x' or 'y' ,default:'y'.
        label-------default label.
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        magnitude---in integrate a vector's module, set True, default:False.
        folder------file path, [folder number,folder start], default:[4,1].
        semiwidth---average semi-width, default:2.
        '''
        import os
        from epoch_class import epoch_class as mysdf
        import numpy as np
        from matplotlib import pyplot as plt
        sc = mysdf()
        col_sty = sc.line_set()
        cwd = os.getcwd()
        s = '/data'
        plt.figure(figsize=(10,5))
        for i in range(folder[0]):
            #set path
            path = cwd + s + str(i+folder[1])
            os.chdir(path)
            rate = sc.reconnection_rate(namelist,field=field,axes=axes,magnitude=magnitude,semiwidth=semiwidth)
            plt.plot(rate,col_sty[i],label=label[0]+label[i+1])
        plt.xlabel('$ t/\omega_{ci}^{-1} $')
        plt.ylabel('$ E_z $')
        plt.title('$ Reconnection $'+'  '+'$ Rate $')
        plt.legend()
        #change to the old directory
        os.chdir(cwd)
        if(display == True):
            plt.show()
        else:
            s1 = 'figure/'
            s2 = 'RECONNECTION_RATE_'
            s3 = axes.upper()
            s4 = '.png'
            path = s1+s2+s3+s4
            plt.savefig(path,dpi=300)
            plt.clf()
#plot ub
    def plot_ub_jb(self,namelist,sub='ub',component=3,factor=1.0,display=True,prefix='1'):
        '''
        This function is used to plot u*b.
        parameters:
        namelist----sdf name list.
        component---x, y, z = 1,2,3 respectively, default:3.
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        prefix------file name prefix, default:1
        '''
        from epoch_class import epoch_class as mysdf
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import numpy as np
        sc = mysdf()
        namelist = sc.get_list(namelist)
        #get sample parameters
        data_dict = sc.get_data(namelist[0],prefix=prefix)
        extent = sc.get_extent(data_dict)
        n = len(namelist)
        narray = []
        for i in range(n):
            if(sub == 'ub'):
                array = sc.cal_ub(namelist[i],component=component)
            if(sub == 'jb'):
                array = sc.cal_jb(namelist[i],component=component)
            narray = narray + [array]
        #find max
        total = np.abs(np.array(narray))
        vmax = total.max()*factor
        #select cmp and vmin
        sample = np.array(narray[0])
        if(np.min(sample) < 0):
            cmap = cm.RdBu_r
            vmin = -vmax
        else:
            cmap = cm.Blues
            vmin = 0
        #plot all the figure
        plt.figure(figsize=(10,5))
        for i in range(n):
            #plt.figure(figsize=(10,5))
            ax = plt.gca()
            #im = ax.contourf(array,100,extent=extent,origin='lower',cmap=cmap)
            im = ax.imshow(narray[i],extent=extent,origin='lower',cmap=cmap,vmax=vmax,vmin=vmin,\
                           interpolation='spline36')
            #add label
            plt.xlabel("$ X/d_i $")
            plt.ylabel("$ Y/d_i $")
            #add title
            t1 = "$ time = "
            t2 = str(namelist[i])
            t3 = "\omega_{ci}^{-1} $"
            sub_index = ['_x $','_y $','_z $']
            if(sub == 'ub'):
                t = r'$ (\vec E + \vec v\times \vec B)'+sub_index[component-1]+"  "+t1+t2+"  "+t3
            if(sub == 'jb'):  
                t = r'$ (\vec j\times \vec B /ne)'+sub_index[component-1]+"  "+t1+t2+"  "+t3
            plt.title(t)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right',size='3%',pad=0.1)
            plt.colorbar(im,cax=cax)
            #select display or save picture
            if(display == True):
                plt.show()
            else:
                s1 = "figure/"
                snumber = str(namelist[i])
                s2 = snumber.zfill(4)
                s3 = ".png"
                path = s1+'Derived_'+sub+'_'+str(component)+'_'+s2+s3
                plt.savefig(path,dpi=300)
                #plt.close()
                plt.clf()
#plot charge density
    def plot_density(self,namelist,display=True,species=3,charge=[1,1,1],factor=1.0,\
                     prefix='1'):
        '''
        This function is used to plot charge density.
        parameters:
        filenumber--sdf file number.
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        species-----pro species.
        charge------electric charge for each species.
        factor------limit factor, default:1.0
        prefix------file name prefix, default:1
        '''
        from epoch_class import epoch_class as mysdf
        from bubble_mr import bubble_mr as const
        import numpy as np
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        sc = mysdf()
        namelist = sc.get_list(namelist)
        #get sample parameters
        data_dict = sc.get_data(namelist[0],prefix=prefix)
        extent = sc.get_extent(data_dict)
        n = len(namelist)
        narray = []
        for eachone in range(n):
            array = sc.charge_density(namelist[eachone],species=species,charge=charge)
            narray = narray + [array]
        total = np.abs(np.array(narray))
        vmax = total.max()*factor
        plt.figure(figsize=(10,5))
        for i in range(n):
            ax = plt.gca()
            #im = ax.contourf(array,100,extent=extent,origin='lower',cmap=cmap)
            im = ax.imshow(narray[i],extent=extent,origin='lower',cmap=cm.RdBu_r,vmax=vmax,vmin=-vmax,
                           interpolation='spline36')
            #add label
            plt.xlabel("$ X/d_i $")
            plt.ylabel("$ Y/d_i $")
            #add title
            t1 = "$ time = "
            t2 = str(namelist[i])
            t3 = "\omega_{ci}^{-1} $"
            t = r'$ charge density $'+"  "+t1+t2+"  "+t3
            plt.title(t)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right',size='3%',pad=0.1)
            plt.colorbar(im,cax=cax)
            #select display or save picture
            if(display == True):
                plt.show()
            else:
                s1 = "figure/"
                snumber = str(namelist[i])
                s2 = snumber.zfill(4)
                s3 = ".png"
                path = s1+'charge_density_'+'_'+s2+s3
                plt.savefig(path,dpi=300)
                #plt.close()
                plt.clf()
#plot dissipation term for each species
    def plot_dissipation_s(self,namelist,display=True,factor=0.5,prefix='1'):
        '''
        This function is used to plot dissipation scalar for each species
        parameters:
        namelist----sdf name list.
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        prefix------file name prefix, default:1
        '''
        from epoch_class import epoch_class as mysdf
        import numpy as np
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        sc = mysdf()
        namelist = sc.get_list(namelist)
        n = len(namelist)
        #species = ['ele','pro1','pro2','pro3']
        #use sample data to get extent
        data_dict = sc.get_data(namelist[0],prefix=prefix)
        extent = sc.get_extent(data_dict)
        title = [r'$ D_{ele} $',r'$ D_{pro1} $',r'$ D_{pro2} $',r'$ D_{pro3} $',r'$ D_{pro} $']
        for i in range(n):
            vector = sc.cal_dissipation_s(namelist[i])
            vmax = np.max(np.array(vector))*factor
            if(np.min(np.array(vector)) <= 0):
                cmap = cm.RdBu_r
                vmin = -vmax
            else:
                cmap = cm.Blues
                vmin = 0
            for j in range(len(vector)):
                plt.figure(i,figsize=(10,5))
                ax = plt.gca()
                #im = ax.contourf(array,100,extent=extent,origin='lower',cmap=cmap)
                im = ax.imshow(vector[j],extent=extent,origin='lower',cmap=cmap,vmax=vmax,vmin=vmin,\
                               interpolation='spline36')
                #add label
                plt.xlabel("$ X/d_i $")
                plt.ylabel("$ Y/d_i $")
                #add title
                t1 = "$ time = "
                t2 = str(namelist[i])
                t3 = "\omega_{ci}^{-1} $"
                t = title[j]+"  "+t1+t2+"  "+t3
                plt.title(t)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right',size='3%',pad=0.1)
                plt.colorbar(im,cax=cax)
                #display or save figure
                if(display == True):
                    plt.show()
                else:
                    s1 = 'figure/dissipation/'
                    s2 = 'Dissipation_s' 
                    s3 = str(namelist[i])+'_'+str(j+1)
                    s4 = '.png'
                    path = s1+s2+s3+s4
                    plt.savefig(path,dpi=300)
                    plt.clf()
#plot energy spectrum
    def plot_enspe(self,namelist,species='ele1',info='momentum',ndomain=1000,prefix='2',\
                   mass_ratio=1,g_max=1.05,display=True,mode=1,average=False,\
                   limx=[-3.0,1.0],limy=[-6.0,0.0]):
        '''
        This function is used to calculate energy spectrum.
        parameters:
        namelist----sdf file name list.
        species-----pro species, default:ele1
        info--------particle information, default:momentum.
        ndomain-----axis step, default:1000.
        prefix------file name prefix, default:2
        mass_ratio--mass ratio to electron, default:1
        g_max-------max gamma, default:1.05
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        mode--------linear or log, default:1
        average-----if average, default:False
        '''
        from epoch_class import epoch_class as mysdf
        import numpy as np
        from matplotlib import pyplot as plt
        sc = mysdf()
        col_sty = sc.line_set()
        namelist = sc.get_list(namelist)
        n = len(namelist)
        nvector = []
        for i in range(n):
            vector = sc.cal_enspe(namelist[i],species=species,info=info,ndomain=ndomain,\
                                  prefix=prefix,mass_ratio=mass_ratio,g_max=g_max,\
                                  average=average)
            nvector = nvector + [vector]
        plt.figure(i,figsize=(10,5))
        for i in range(n):
            plt.plot(nvector[i][0],nvector[i][1],col_sty[i],label='time = '+str(namelist[i]))
        plt.xlabel('$ \gamma $')
        plt.ylabel('Number of particle')
        plt.title('Energy spectrum ' + species)
        plt.legend()
        #mode
        if(mode == 1):
            pass
            plt.xlim(xmin=10**limx[0],xmax=10**limx[1])
            plt.ylim(ymin=10**limy[0],ymax=10**limy[1])
        elif(mode == 2):
            plt.xlim(xmin=10**limx[0],xmax=10**limx[1])
            plt.ylim(ymin=10**limy[0],ymax=10**limy[1])
            plt.yscale('log')
        else:
            plt.xlim(xmin=10**limx[0],xmax=10**limx[1])
            plt.ylim(ymin=10**limy[0],ymax=10**limy[1])
            plt.xscale('log')
            plt.yscale('log')
        if(display == True):
            plt.show()
        else:
            s1 = 'figure/'
            s2 = 'energy_spectrum_'+species
            s3 = '.png'
            path = s1+s2+s3
            plt.savefig(path,dpi=300)
            plt.clf()
        #return (nvector[n-1][0],nvector[n-1][1])
#plot energy spectrum of differrent cases
    def plot_enspe_s(self,namelist,species='ele1',info='momentum',ndomain=1000,prefix='2',\
                   mass_ratio=1,g_max=1.05,display=True,mode=1,average=False,\
                   limx=[-3.0,1.0],limy=[-6.0,0.0],files=[1,2,3]):
        '''
        This function is used to calculate energy spectrum.
        parameters:
        namelist----sdf file name list.
        species-----pro species, default:ele1
        info--------particle information, default:momentum.
        ndomain-----axis step, default:1000.
        prefix------file name prefix, default:2
        mass_ratio--mass ratio to electron, default:1
        g_max-------max gamma, default:1.05
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        mode--------linear or log, default:1
        average-----if average, default:False
        files-------file path
        '''
        from epoch_class import epoch_class as mysdf
        import numpy as np
        from matplotlib import pyplot as plt
        import os
        path = os.getcwd()
        sc = mysdf()
        col_sty = sc.line_set()
        namelist = sc.get_list(namelist)
        n = len(files)
        nvector = []
        for i in range(len(files)):
            os.chdir(path+'/data'+str(files[i]))
            vector = sc.cal_enspe(namelist[0],species=species,info=info,ndomain=ndomain,\
                                  prefix=prefix,mass_ratio=mass_ratio,g_max=g_max,\
                                  average=average)
            nvector = nvector + [vector]
        plt.figure(i,figsize=(10,5))
        labels=['100:100','400:400','25:800','25:1600']
        for i in range(n):
            plt.plot(nvector[i][0],nvector[i][1],col_sty[i],label='mass_ratio'+' = '+labels[i])
        plt.xlabel('$ \gamma - 1$')
        plt.ylabel('Number of particle')
        plt.title('Energy spectrum ' + species)
        plt.legend()
        #mode
        if(mode == 1):
            pass
            plt.xlim(xmin=10**limx[0],xmax=10**limx[1])
            plt.ylim(ymin=10**limy[0],ymax=10**limy[1])
        elif(mode == 2):
            plt.xlim(xmin=10**limx[0],xmax=10**limx[1])
            plt.ylim(ymin=10**limy[0],ymax=10**limy[1])
            plt.yscale('log')
        else:
            plt.xlim(xmin=10**limx[0],xmax=10**limx[1])
            plt.ylim(ymin=10**limy[0],ymax=10**limy[1])
            plt.xscale('log')
            plt.yscale('log')
        if(display == True):
            plt.show()
        else:
            s1 = 'figure/'
            s2 = 'energy_spectrum_'+species
            s3 = '.png'
            path = s1+s2+s3
            plt.savefig(path,dpi=300)
            plt.clf()
#implot 3d
    def implot_3d(self,filenumber=0,field='ey',display=True,factor=1.0,prefix='1',\
                  average=False, nslices=1, axis='x',index=0,nspe=3, dpix=10, dpiy=5):
        '''
        This function is used to visualize 2d field data.
        parameters:
        filenumber--sdf file number, an integer, default:0.
        field-------physical field, default:'bx'.
        display-----if want to display figure, set True, if not, the figure will be save to a png file.default:True.
        factor------vmax multifactor,less than 1, default:1.0
        prefix------file name prefix, default:1
        average-----if average on axis.
        nslices-----average onver n slices.
        axis--------slice axis, default:'x'
        index-------slice index, default:0
        dpix--------figure size x.
        dpiy--------figure size y.
        '''
        import numpy as np
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from epoch_class import epoch_class as sdfclass
        sc = sdfclass()
        #get name list
        namelist = sc.get_list(filenumber)
        n = len(namelist)
        #get sample parameters
        data_dict = sc.get_data(namelist[0],prefix=prefix)
        if(field != 'sigma'):
            if (field == 'charge_density'):
                constant = sc.get_constant_3d(field='Derived_Number_Density_ele1')
            else:
                keys = data_dict.keys()
                field = sc.get_field(field=field,keys=keys)
                constant = sc.get_constant_3d(field=field)
        extent = sc.get_extent_3d(data_dict,axis=axis)
        #for loop to get array from every sdf file
        if(average == False):
            k = 0
        else:
            k = nslices
        narray = []
        for eachone in namelist:
            t_array = 0
            if(field == 'sigma'):
                t_array = sc.cal_sigma(eachone,species='ele2',index=index,axis=axis,prefix=prefix)
            else:
                data_dict = sc.get_data(eachone)
                for j in range(index-k, index+k+1):
                    array = sc.get_array_3d(data_dict,field=field,axis=axis,index=j,nspe=nspe)/constant
                    t_array = t_array + array
            narray = narray + [t_array/(2.0*k+1.0)]
        #find max
        total = np.abs(np.array(narray))
        vmax = total.max()*factor
        #select cmp and vmin
        sample = np.array(narray[0])
        #[a, b] = sample.shape
        #a = int(a/100*2)
        #b = int(b/100*2)
        if(np.min(sample) < 0):
            cmap = cm.RdBu_r
            vmin = -vmax
        else:
            cmap = cm.Blues
            vmin = 0
        #plot all the figure
        plt.figure(figsize=(dpix, dpiy))
        for i in range(n):
            ax = plt.gca()
            #im = ax.contourf(array,100,extent=extent,origin='lower',cmap=cmap)
            im = ax.imshow(narray[i],extent=extent,origin='lower',cmap=cmap,vmax=vmax,vmin=vmin,\
                           interpolation='spline36')
            #add label
            cordi = [r"$ X/\lambda_0 $",r"$ Y/\lambda_0 $",r"$ Z/\lambda_0 $"]
            if(axis == 'x'):
                plt.xlabel(cordi[1])
                plt.ylabel(cordi[2])
            elif(axis == 'y'):
                plt.xlabel(cordi[0])
                plt.ylabel(cordi[2])
            else:
                plt.xlabel(cordi[0])
                plt.ylabel(cordi[1])
            #add title
            t1 = "$ time = "
            t2 = str(namelist[i])
            t3 = " T $"
            t = field+"  "+t1+t2+"  "+t3
            plt.title(t)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right',size='3%',pad=0.1)
            plt.colorbar(im,cax=cax)
            #select display or save picture
            if(display == True):
                plt.show()
            else:
                s1 = "figure/"
                snumber = str(namelist[i])
                s2 = snumber.zfill(4)
                s3 = ".png"
                path = s1+field+s2+"_"+axis+str(index)+s3
                plt.savefig(path,dpi=300)
                #plt.close()
                plt.clf()
#plot particles information
    def plot_parinfo(self,filenumber,field='p',species='ele',prefix=3,component=1,\
                     id_index=1,mass_ratio=1,display=True,time_factor=1.0, case='laser'):
        '''
        This function is used to plot particle information.
        parameters:
        filenumber--sdf file number.
        field-------physical field, default:Px.
        species-----species, default:ele
        prefix------file name prefix, default:3
        component---axis, default:1(x)
        id_index----particle id.
        mass_ratio--mass ratio, default, default:1
        display-----display figure, default:True
        time_factor-time factor, an output sdf file represent time step, default:1
        '''
        from epoch_class import epoch_class as mysdf
        from matplotlib import pyplot as plt
        import numpy as np
        #constants
        from constants import laser_mr as const
        if(case == 'laser'):
            from constants import laser_mr as const
            if(field == 'p'):
                norm = const.P0 * mass_ratio
            else:
                norm = const.la
            x_label = '$ t/T_0 $'
        else:
            from constants import bubble_mr as const
            if(field == 'p'):
                norm = const.Pe0 * mass_ratio
            else:
                norm = const.di
            x_label = '$ t/\omega_{ci}^{-1} $'
        sc = mysdf()
        namelist = sc.get_list(filenumber)
        id_index = sc.get_list(id_index)
        n = len(namelist)
        axis = np.array(namelist)*time_factor
        #len_f = len(field)
        #len_c = len(component)
        if(field == 'p'):
            fields = ['Px','Py','Pz']
            label = fields
            components = [1,2,3]
            ylabel = 'Momentum'
        else:
            fields = ['Grid','Grid','Grid']
            label = ['X','Y','Z']
            components = [1,2,3]
            ylabel = 'Position'
        len_c = len(components)
        len_id = len(id_index)
        plt.figure(figsize=(10,5))
        for k in range(len_id):
            narray = []
            for i in range(len_c):
                array = sc.get_parinfo(namelist,field=fields[i],species=species, \
                                       component=components[i],id_index=id_index[k],\
                                       mass_ratio=mass_ratio,time_factor=time_factor)
                narray += [array]
            color_set = sc.line_set()
           #plt.figure(figsize=(10,5))
            for j in range(len_c):
                plt.plot(axis,narray[j]/norm,color_set[j],label=label[j])
                plt.xlabel(x_label)
                plt.ylabel(ylabel)
                plt.title(species+'_'+str(id_index[k]))
                plt.legend()
            if(display == True):
                plt.show()
            else:
                s1 = "figure/parinfo/"
                s2 = ".png"
                path = s1+field+'_'+species+'_'+str(id_index[k])+s2
                plt.savefig(path,dpi=300)
                #plt.close()
                plt.clf()  
#plot particle information in 3d
    def plot_parinfo_3d(self,filenumber,field='Pz',species='ele1',prefix='2',component=3,mass_ratio=1,display=True,least=2.5,limx=[20,30],limy=[-5,5],limz=[-6,6],data_range=[-15,15],cord=True,ifreturn=False):
        '''
        This function is used to plot particle information.
        parameters:
        filenumber--sdf file number.
        field-------physical field, default:Px.
        species-----species, default:ele
        prefix------file name prefix, default:3
        component---axis, default:1(x)
        mass_ratio--mass ratio, default, default:1
        display-----display figure, default:True
        least-------below the least, not show, default:2.5.
        data_range--data range, default:[-15,15]
        cord--------if grid, default:True
        '''
        import numpy as np
        import matplotlib
        import matplotlib.cm as cmx
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from epoch_class import epoch_class as mysdf
        sc = mysdf()
        namelist = sc.get_list(filenumber)
        n = len(namelist)
        narray = []
        fig = plt.figure(figsize=(10,5))
        for i in range(n):
            #get grid
            grid = sc.get_scatter(namelist[i],field='Grid',species=species,prefix=prefix,component=component,mass_ratio=mass_ratio)
            x = grid[0]
            y = grid[1]
            z = grid[2]
            #get particles information
            data = sc.get_scatter(namelist[i],field=field,species=species,prefix=prefix,component=component,mass_ratio=mass_ratio)
            p = 0
            for j in range(len(data)):
                p = p + data[j]*data[j]
            p = np.sqrt(p)
            fields = ['Px','Py','Pz']
            if(field[3:] == 'd'):
                for k in range(len(fields)):
                    if(field[:2] == fields[k]):
                        p_d = data[k]/p
                result = sc.select_data(x=x,y=y,z=z,data=p_d,base=p,least=least,cord=cord)
            else:
                for k in range(len(fields)):
                    if(field == fields[k]):
                        p_c = data[k]
                result = sc.select_data(x=x,y=y,z=z,data=p_c,base=p,least=least,cord=cord)
            x = result[0]
            y = result[1]
            z = result[2]
            data = result[3]
            #plot the figure
            ax = fig.add_subplot(111,projection='3d')
            cm = plt.get_cmap('jet')
            cNorm = matplotlib.colors.Normalize(vmin=data_range[0],vmax=data_range[1]);
            scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cm)
            gci = ax.scatter(x,y,z,c=scalarMap.to_rgba(data),edgecolor='none')
            scalarMap.set_array(data);
            #fig.colorbar(scalarMap)
            cbar = plt.colorbar(scalarMap)
            ax.set_title('$ time = '+str(namelist[i])+' T_0 $')
            ax.set_xlabel('$ X/\lambda_0 $')
            ax.set_ylabel('$ Y/\lambda_0 $')
            ax.set_zlabel('$ Z/\lambda_0 $')
            ax.set_xlim(limx[0],limx[1])
            ax.set_ylim(limy[0],limy[1])
            ax.set_zlim(limz[0],limz[1])
            if(field[3:] == 'd'):
                cbar.set_label('$ P_y/P $')
            else:
                cbar.set_label('$ P_z/(m_ec) $')
            if(display == True):
                plt.show()
            else:
                s1 = 'figure/'
                s2 = '.png'
                s3 = species+'_'+field+'_'+str(namelist[i])
                path = s1 + s3 + s2
                plt.savefig(path,dpi=300)
                #plt.close()
                plt.clf()
            if(ifreturn == True):
                return np.sum(np.abs(data))/float(len(data))
#plot magnetic field tension force in 3d
    def plot_tensor_force_3d(self,filenumber,component=3,prefix='1',index=75,axis='y',\
                             display=True,factor=1.0,axis_lim=[24,-4,27,4],data_range=24):
        '''
        This function is used to plot magnetic tension force in 3d.
        parameters:
        filenumber--sdf file number.
        prefix------file name prefix, default:3
        component---axis, default:3(z)
        index-------slice index, default:75.
        axis--------axis, default:'y'.
        display-----display figure, default:True
        '''
        import numpy as np
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from epoch_class import epoch_class as mysdf
        from constants import laser_mr as const
        sc = mysdf()
        #get name list
        namelist = sc.get_list(filenumber)
        n = len(namelist)
        #get sample parameters
        data_dict = sc.get_data(namelist[0],prefix=prefix)
        extent = sc.get_extent_3d(data_dict,axis=axis)
        narray = []
        for i in range(n):
            array = sc.tensor_force_3d(namelist[i],component=component,prefix=prefix,index=index,axis=axis)
            array = np.transpose(array)/const.Pa0
            narray = narray + [array]
        #find max
        total = np.abs(np.array(narray))
        #vmax = total.max()*factor
        #if(data_range < vmax):
        vmax = data_range
        #select cmp and vmin
        sample = np.array(narray[0])
        if(np.min(sample) < 0):
            cmap = cm.jet
            vmin = -vmax
        else:
            cmap = cm.Blues
            vmin = 0
        #plot all the figure
        plt.figure(figsize=(10,5))
        for i in range(n):
            ax = plt.gca()
            #im = ax.contourf(array,100,extent=extent,origin='lower',cmap=cmap)
            im = ax.imshow(narray[i],extent=extent,origin='lower',cmap=cmap,vmax=vmax,vmin=vmin,\
                           interpolation='spline36')
            #add label
            if(axis == 'x'):
                plt.xlabel("$ Y/\lambda_0 $")
                plt.ylabel("$ Z/\lambda_0 $")
            elif(axis == 'y'):
                plt.xlabel("$ X/\lambda_0 $")
                plt.ylabel("$ Z/\lambda_0 $")
            else:
                plt.xlabel("$ X/\lambda_0 $")
                plt.ylabel("$ Y/\lambda_0 $")
            #add title
            t1 = "$ time = "
            t2 = str(namelist[i])
            t3 = "T_0 $"
            t = "Magnetic_tension_force "+t1+t2+"  "+t3
            plt.title(t)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right',size='3%',pad=0.1)
            cbar = plt.colorbar(im,cax=cax)
            ax.set_xlim(axis_lim[0],axis_lim[2])
            ax.set_ylim(axis_lim[1],axis_lim[3])
            cbar.set_label('$ T^m_z/(10^{19}Pa/m) $')
            #select display or save picture
            if(display == True):
                plt.show()
            else:
                s1 = "figure/"
                snumber = str(namelist[i])
                s2 = snumber.zfill(4)
                s3 = ".png"
                path = s1+'Tension_force_'+axis+str(index)+s2+s3
                plt.savefig(path,dpi=300)
                #plt.close()
                plt.clf()
#plot energy
    def plot_energy(self,filenumber,info='gamma',dimen=3,frac=0.5,display=True):
        '''
        This function is used to plot magnetic tension force in 3d.
        parameters:
        filenumber--sdf file filenumber.
        info--------particle information, default:'gamma'
        dimen-------dimensions, default:3
        frac--------output fraction, default:0.5
        display-----display figure, default:True
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from epoch_class import epoch_class as mysdf
        sc = mysdf()
        col_sty = sc.line_set()
        #get name list
        namelist = sc.get_list(filenumber)
        n = len(namelist)
        mag_en_a = np.zeros(n,np.float)
        ele_en_a = np.zeros(n,np.float)
        mag_en = np.zeros(n,np.float)
        ele_en = np.zeros(n,np.float)
        par_ele1 = np.zeros(n,np.float)
        par_ele2 = np.zeros(n,np.float)
        par_pro1 = np.zeros(n,np.float)
        par_pro2 = np.zeros(n,np.float)
        for i in range(n):
            #mag_en[i] = 1000*sc.cal_field_en(namelist[i],dimen=dimen,prefix='1',field='magnetic',average=False)
            #ele_en[i] = 1000*sc.cal_field_en(namelist[i],dimen=dimen,prefix='1',field='electric',average=False)
            mag_en_a[i] = 1000*sc.cal_field_en(namelist[i],dimen=dimen,prefix='1',field='magnetic',average=True)
            ele_en_a[i] = 1000*sc.cal_field_en(namelist[i],dimen=dimen,prefix='1',field='electric',average=True)
            #par_ele1[i] = 1000*sc.cal_particle_en(namelist[i],prefix='2',species='ele1',info=info,frac=frac,mass_ratio=1)
            #par_ele2[i] = 1000*sc.cal_particle_en(namelist[i],prefix='2',species='ele2',info=info,frac=frac,mass_ratio=1)
            #par_pro1[i] = 1000*sc.cal_particle_en(namelist[i],prefix='2',species='pro1',info=info,frac=frac,mass_ratio=1836)
            #par_pro2[i] = 1000*sc.cal_particle_en(namelist[i],prefix='2',species='pro2',info=info,frac=frac,mass_ratio=1836)
        #calculate energy loss and gain
        #en_gain = par_ele2 + par_pro2 + ele_en_a
        #en_loss = ele_en + mag_en - ele_en_a - mag_en_a + par_ele1
        #en_m = mag_en_a
        #en_gain = np.array(en_gain - en_gain[0])
        #en_loss = np.array(en_loss - en_loss[0])
        #en_m = np.array(en_m - en_m[0])
        #return (en_gain,en_loss,en_m)
        #plot
        plt.figure(figsize=(10, 5))
        plt.plot(namelist,mag_en_a,col_sty[1],label='$ E_{magnetic}$')
        plt.plot(namelist,ele_en_a,col_sty[2],label='$ E_{electric}$')
        #plt.plot(namelist,par_ele1,col_sty[3],label='$ E_{ele1} $')
        #plt.plot(namelist,par_ele2,col_sty[4],label='$ E_{ele2} $')
        plt.xlabel('$ t/T_0 $')
        plt.ylabel('E/(mJ)')
        plt.legend()
        if(display == True):
            plt.show()
        else:
            s1 = 'figure/'
            s4 = '.png'
            path = s1+'Energy'+s4
            plt.savefig(path,dpi=300)
            plt.clf()
        #return (en_gain,en_loss,en_m)

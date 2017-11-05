#include <iostream>
#include <stdio.h>
#include <math.h>

#include "mail.h"
#include "io_flow.h"
#include "utils.h"

using namespace std;

vector<float> flowErrorsOutlier (FlowImage &F_gt,FlowImage &F_orig,FlowImage &F_ipol) {

  // check file size
  if (F_gt.width()!=F_orig.width() || F_gt.height()!=F_orig.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }

  // extract width and height
  int32_t width  = F_gt.width();
  int32_t height = F_gt.height();

  // init errors
  vector<float> errors;
  for (int32_t i=0; i<2*5; i++)
    errors.push_back(0);
  int32_t num_pixels = 0;
  int32_t num_pixels_result = 0;

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {
      float fu = F_gt.getFlowU(u,v)-F_ipol.getFlowU(u,v);
      float fv = F_gt.getFlowV(u,v)-F_ipol.getFlowV(u,v);
      float f_err = sqrt(fu*fu+fv*fv);
      if (F_gt.isValid(u,v)) {
        for (int32_t i=0; i<5; i++)
          if (f_err>(float)(i+1))
            errors[i*2+0]++;
        num_pixels++;
        if (F_orig.isValid(u,v)) {
          for (int32_t i=0; i<5; i++)
            if (f_err>(float)(i+1))
              errors[i*2+1]++;
          num_pixels_result++;
        }
      }
    }
  }

  // check number of pixels
  if (num_pixels==0) {
    cout << "ERROR: Ground truth defect => Please write me an email!" << endl;
    throw 1;
  }

  // normalize errors
  for (int32_t i=0; i<errors.size(); i+=2)
    errors[i] /= max((float)num_pixels,1.0f);
  if (num_pixels_result>0)
    for (int32_t i=1; i<errors.size(); i+=2)
      errors[i] /= max((float)num_pixels_result,1.0f);

  // push back density
  errors.push_back((float)num_pixels_result/max((float)num_pixels,1.0f));

  // return errors
  return errors;
}

vector<float> flowErrorsAverage (FlowImage &F_gt,FlowImage &F_orig,FlowImage &F_ipol) {

  // check file size
  if (F_gt.width()!=F_orig.width() || F_gt.height()!=F_orig.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }

  // extract width and height
  int32_t width  = F_gt.width();
  int32_t height = F_gt.height();

  // init errors
  vector<float> errors;
  for (int32_t i=0; i<2; i++)
    errors.push_back(0);
  int32_t num_pixels = 0;
  int32_t num_pixels_result = 0;

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {
      float fu = F_gt.getFlowU(u,v)-F_ipol.getFlowU(u,v);
      float fv = F_gt.getFlowV(u,v)-F_ipol.getFlowV(u,v);
      float f_err = sqrt(fu*fu+fv*fv);
      if (F_gt.isValid(u,v)) {
        errors[0] += f_err;
        num_pixels++;
        if (F_orig.isValid(u,v)) {
          errors[1] += f_err;
          num_pixels_result++;
        }
      }
    }
  }

  // normalize errors
  errors[0] /= max((float)num_pixels,1.0f);
  errors[1] /= max((float)num_pixels_result,1.0f);

  // return errors
  return errors;
}

void plotVectorField (FlowImage &F,string dir,char* prefix) {

  // command for execution
  char command[1024];
  
  // write flow field to file
  FILE *fp = fopen((dir + prefix + ".txt").c_str(),"w");
  for (int32_t u=5; u<F.width()-5; u+=20) {
    for (int32_t v=5; v<F.height()-5; v+=20) {
      fprintf(fp,"%d %d %f %f\n",u,v,F.getFlowU(u,v),F.getFlowV(u,v));
    }
  }
  fclose(fp);
  
  // create png + eps
  for (int32_t j=0; j<2; j++) {

    // open file  
    FILE *fp = fopen((dir + prefix + ".gp").c_str(),"w");

    // save gnuplot instructions
    if (j==0) {
      fprintf(fp,"set term png size %d,%d font \"Helvetica\" 11\n",F.width()*2,F.height()*2);
      fprintf(fp,"set output \"%s.png\"\n",prefix);
    } else {
      fprintf(fp,"set term postscript eps enhanced color\n");
      fprintf(fp,"set output \"%s.eps\"\n",prefix);
      fprintf(fp,"set size ratio -1\n");
    }

    // plot options (no borders)    
    fprintf(fp,"set yrange [%d:0] reverse\n",F.height());
    fprintf(fp,"set lmargin 0\n");
    fprintf(fp,"set bmargin 0\n");
    fprintf(fp,"set rmargin 0\n");
    fprintf(fp,"set tmargin 0\n");
    fprintf(fp,"set noxtic\n");
    fprintf(fp,"set noytic\n");
    fprintf(fp,"set nokey\n");
    
    // plot error curve
    fprintf(fp,"plot \"%s.txt\" using 1:2:3:4 lc 0 w vector",prefix);
    
    // close file
    fclose(fp);
    
    // run gnuplot => create png + eps
    sprintf(command,"cd %s; gnuplot %s",dir.c_str(),((string)prefix + ".gp").c_str());
    system(command);
    
    // resize png to original size, using aliasing
    if (j==0) {
      sprintf(command,"mogrify -resize %dx%d %s",F.width(),F.height(),(dir + prefix + ".png").c_str());
      system(command);
    }
  }
  
  // create pdf and crop
  sprintf(command,"cd %s; ps2pdf %s.eps %s_large.pdf",dir.c_str(),prefix,prefix);
  system(command);
  sprintf(command,"cd %s; pdfcrop %s_large.pdf %s.pdf",dir.c_str(),prefix,prefix);
  system(command);
  sprintf(command,"cd %s; rm %s_large.pdf",dir.c_str(),prefix);
  system(command);
}

bool eval (string result_sha,Mail* mail) {

  // ground truth and result directories
  string gt_noc_dir = "data/stereo_flow/flow_noc";
  string gt_occ_dir = "data/stereo_flow/flow_occ";
  string gt_img_dir = "data/stereo_flow/image_0";
  string result_dir = "results/" + result_sha;

  // create output directories
  system(("mkdir " + result_dir + "/errors_noc_out/").c_str());
  system(("mkdir " + result_dir + "/errors_occ_out/").c_str());
  system(("mkdir " + result_dir + "/errors_noc_avg/").c_str());
  system(("mkdir " + result_dir + "/errors_occ_avg/").c_str());
  system(("mkdir " + result_dir + "/errors_img/").c_str());
  system(("mkdir " + result_dir + "/flow_orig/").c_str());
  system(("mkdir " + result_dir + "/flow_ipol/").c_str());
  system(("mkdir " + result_dir + "/flow_field/").c_str());
  system(("mkdir " + result_dir + "/image_0/").c_str());

  // vector for storing the errors
  vector< vector<float> > errors_noc_out;
  vector< vector<float> > errors_occ_out;
  vector< vector<float> > errors_noc_avg;
  vector< vector<float> > errors_occ_avg;

  // for all test files do
  for (int32_t i=0; i<195; i++) {

    // file name
    char prefix[256];
    sprintf(prefix,"%06d_10",i);
    
    // output
    mail->msg("Processing: %s.png",prefix);

    // catch errors, when loading fails
    try {

      // load ground truth flow maps
      FlowImage F_gt_noc(gt_noc_dir + "/" + prefix + ".png");
      FlowImage F_gt_occ(gt_occ_dir + "/" + prefix + ".png");

      // check submitted result
      string image_file = result_dir + "/data/" + prefix + ".png";
      if (!imageFormat(image_file,png::color_type_rgb,16,F_gt_noc.width(),F_gt_noc.height())) {
        mail->msg("ERROR: Input must be png, 3 channels, 16 bit, %d x %d px",
                  F_gt_noc.width(),F_gt_noc.height());
        return false;        
      }
      
      // load submitted result
      FlowImage F_orig(image_file);
      
      // interpolate missing values
      FlowImage F_ipol(F_orig); 
      F_ipol.interpolateBackground();     

      // add flow errors
      vector<float> errors_noc_out_curr = flowErrorsOutlier(F_gt_noc,F_orig,F_ipol);
      vector<float> errors_occ_out_curr = flowErrorsOutlier(F_gt_occ,F_orig,F_ipol);
      vector<float> errors_noc_avg_curr = flowErrorsAverage(F_gt_noc,F_orig,F_ipol);
      vector<float> errors_occ_avg_curr = flowErrorsAverage(F_gt_occ,F_orig,F_ipol);
      errors_noc_out.push_back(errors_noc_out_curr);
      errors_occ_out.push_back(errors_occ_out_curr);
      errors_noc_avg.push_back(errors_noc_avg_curr);
      errors_occ_avg.push_back(errors_occ_avg_curr);

      // save detailed infos for first 20 images
      if (i<20) {

        // save errors of error images to text file
        FILE *errors_noc_out_file = fopen((result_dir + "/errors_noc_out/" + prefix + ".txt").c_str(),"w");
        FILE *errors_occ_out_file = fopen((result_dir + "/errors_occ_out/" + prefix + ".txt").c_str(),"w");
        FILE *errors_noc_avg_file = fopen((result_dir + "/errors_noc_avg/" + prefix + ".txt").c_str(),"w");
        FILE *errors_occ_avg_file = fopen((result_dir + "/errors_occ_avg/" + prefix + ".txt").c_str(),"w");
        if (errors_noc_out_file==NULL || errors_occ_out_file==NULL ||
            errors_noc_avg_file==NULL || errors_occ_avg_file==NULL) {
          mail->msg("ERROR: Couldn't generate/store output statistics!");
          return false;
        }
        for (int32_t j=0; j<errors_noc_out_curr.size(); j++) {
          fprintf(errors_noc_out_file,"%f ",errors_noc_out_curr[j]);
          fprintf(errors_occ_out_file,"%f ",errors_occ_out_curr[j]);
        }
        for (int32_t j=0; j<errors_noc_avg_curr.size(); j++) {
          fprintf(errors_noc_avg_file,"%f ",errors_noc_avg_curr[j]);
          fprintf(errors_occ_avg_file,"%f ",errors_occ_avg_curr[j]);
        }
        fclose(errors_noc_out_file);
        fclose(errors_occ_out_file);
        fclose(errors_noc_avg_file);
        fclose(errors_occ_avg_file);

        // save error images
        png::image<png::rgb_pixel> F_err = F_ipol.errorImage(F_gt_noc,F_gt_occ);
        F_err.write(result_dir + "/errors_img/" + prefix + ".png");

        // find maximum ground truth flow
        float max_flow = F_gt_occ.maxFlow();

        // save original flow image
        F_orig.writeColor(result_dir + "/flow_orig/" + prefix + ".png",max_flow);
        
        // save interpolated flow image
        F_ipol.writeColor(result_dir + "/flow_ipol/" + prefix + ".png",max_flow);
        
        // save interpolated flow vector field
        plotVectorField(F_ipol,result_dir + "/flow_field/",prefix);

        // copy left camera image        
        string img_src = gt_img_dir + "/" + prefix + ".png";
        string img_dst = result_dir + "/image_0/" + prefix + ".png";
        system(("cp " + img_src + " " + img_dst).c_str());
      }

    // on error, exit
    } catch (...) {
      mail->msg("ERROR: Couldn't read: %s.png",prefix);
      return false;
    }
  }

  // open stats file for writing
  string stats_noc_out_file_name = result_dir + "/stats_noc_out.txt";
  string stats_occ_out_file_name = result_dir + "/stats_occ_out.txt";
  string stats_noc_avg_file_name = result_dir + "/stats_noc_avg.txt";
  string stats_occ_avg_file_name = result_dir + "/stats_occ_avg.txt";
  FILE *stats_noc_out_file = fopen(stats_noc_out_file_name.c_str(),"w");
  FILE *stats_occ_out_file = fopen(stats_occ_out_file_name.c_str(),"w");
  FILE *stats_noc_avg_file = fopen(stats_noc_avg_file_name.c_str(),"w");
  FILE *stats_occ_avg_file = fopen(stats_occ_avg_file_name.c_str(),"w");
  if (stats_noc_out_file==NULL || stats_occ_out_file==NULL || errors_noc_out.size()==0 || errors_occ_out.size()==0 ||
      stats_noc_avg_file==NULL || stats_occ_avg_file==NULL || errors_noc_avg.size()==0 || errors_occ_avg.size()==0) {
    mail->msg("ERROR: Couldn't generate/store output statistics!");
    return false;
  }
  
  // write mean
  for (int32_t i=0; i<errors_noc_out[0].size(); i++) {
    fprintf(stats_noc_out_file,"%f ",statMean(errors_noc_out,i));
    fprintf(stats_occ_out_file,"%f ",statMean(errors_occ_out,i));
  }
  for (int32_t i=0; i<errors_noc_avg[0].size(); i++) {
    fprintf(stats_noc_avg_file,"%f ",statMean(errors_noc_avg,i));
    fprintf(stats_occ_avg_file,"%f ",statMean(errors_occ_avg,i));
  }
  fprintf(stats_noc_out_file,"\n");
  fprintf(stats_occ_out_file,"\n");
  fprintf(stats_noc_avg_file,"\n");
  fprintf(stats_occ_avg_file,"\n");
  
  // write min
  for (int32_t i=0; i<errors_noc_out[0].size(); i++) {
    fprintf(stats_noc_out_file,"%f ",statMin(errors_noc_out,i));
    fprintf(stats_occ_out_file,"%f ",statMin(errors_occ_out,i));
  }
  for (int32_t i=0; i<errors_noc_avg[0].size(); i++) {
    fprintf(stats_noc_avg_file,"%f ",statMin(errors_noc_avg,i));
    fprintf(stats_occ_avg_file,"%f ",statMin(errors_occ_avg,i));
  }
  fprintf(stats_noc_out_file,"\n");
  fprintf(stats_occ_out_file,"\n");
  fprintf(stats_noc_avg_file,"\n");
  fprintf(stats_occ_avg_file,"\n");
  
  // write max
  for (int32_t i=0; i<errors_noc_out[0].size(); i++) {
    fprintf(stats_noc_out_file,"%f ",statMax(errors_noc_out,i));
    fprintf(stats_occ_out_file,"%f ",statMax(errors_occ_out,i));
  }
  for (int32_t i=0; i<errors_noc_avg[0].size(); i++) {
    fprintf(stats_noc_avg_file,"%f ",statMax(errors_noc_avg,i));
    fprintf(stats_occ_avg_file,"%f ",statMax(errors_occ_avg,i));
  }
  fprintf(stats_noc_out_file,"\n");
  fprintf(stats_occ_out_file,"\n");
  fprintf(stats_noc_avg_file,"\n");
  fprintf(stats_occ_avg_file,"\n");
  
  // close files
  fclose(stats_noc_out_file);
  fclose(stats_occ_out_file);
  fclose(stats_noc_avg_file);
  fclose(stats_occ_avg_file);

  // success
	return true;
}

int32_t main (int32_t argc,char *argv[]) {

  // we need 2 or 4 arguments!
  if (argc!=2 && argc!=4) {
    cout << "Usage: ./eval_flow result_sha [user_sha email]" << endl;
    return 1;
  }

  // read arguments
  string result_sha = argv[1];
  
  // init notification mail
  Mail *mail;
  if (argc==4) mail = new Mail(argv[3]);
  else         mail = new Mail();
  mail->msg("Thank you for participating in our evaluation!");

  // run evaluation
  bool success = eval(result_sha,mail);
  if (argc==4) mail->finalize(success,"flow",result_sha,argv[2]);
  else         mail->finalize(success,"flow",result_sha);

  // send mail and exit
  delete mail;
  return 0;
}


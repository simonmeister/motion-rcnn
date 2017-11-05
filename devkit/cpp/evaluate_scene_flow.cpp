#include <iostream>
#include <stdio.h>
#include <math.h>

#include "mail.h"
#include "io_disp.h"
#include "io_flow.h"
#include "io_integer.h"
#include "utils.h"

#define NUM_TEST_IMAGES 200
#define NUM_ERROR_IMAGES 20
#define ABS_THRESH 3.0
#define REL_THRESH 0.05

using namespace std;

vector<float> disparityErrorsOutlier (DisparityImage &D_gt,DisparityImage &D_orig,DisparityImage &D_ipol,IntegerImage &O_map) {

  // check file size
  if (D_gt.width()!=D_orig.width() || D_gt.height()!=D_orig.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }

  // extract width and height
  int32_t width  = D_gt.width();
  int32_t height = D_gt.height();

  // init errors
  vector<float> errors;
  int32_t num_errors_bg = 0;
  int32_t num_pixels_bg = 0;
  int32_t num_errors_bg_result = 0;
  int32_t num_pixels_bg_result = 0;
  int32_t num_errors_fg = 0;
  int32_t num_pixels_fg = 0;
  int32_t num_errors_fg_result = 0;
  int32_t num_pixels_fg_result = 0;
  int32_t num_errors_all = 0;
  int32_t num_pixels_all = 0;
  int32_t num_errors_all_result = 0;
  int32_t num_pixels_all_result = 0;

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {
      if (D_gt.isValid(u,v)) {
        float d_gt  = D_gt.getDisp(u,v);
        float d_est = D_ipol.getDisp(u,v);
        bool  d_err = fabs(d_gt-d_est)>ABS_THRESH && fabs(d_gt-d_est)/fabs(d_gt)>REL_THRESH;
        if (O_map.getValue(u,v)==0) {
          if (d_err)
            num_errors_bg++;
          num_pixels_bg++;
          if (D_orig.isValid(u,v)) {
            if (d_err)
              num_errors_bg_result++;
            num_pixels_bg_result++;
          }
        } else {
          if (d_err)
            num_errors_fg++;
          num_pixels_fg++;
          if (D_orig.isValid(u,v)) {
            if (d_err)
              num_errors_fg_result++;
            num_pixels_fg_result++;
          }
        }
        if (d_err)
          num_errors_all++;
        num_pixels_all++;
        if (D_orig.isValid(u,v)) {
          if (d_err)
            num_errors_all_result++;
          num_pixels_all_result++;
        }
      }
    }
  }

  // push back errors and pixel count
  errors.push_back(num_errors_bg);
  errors.push_back(num_pixels_bg);
  errors.push_back(num_errors_bg_result);
  errors.push_back(num_pixels_bg_result);
  errors.push_back(num_errors_fg);
  errors.push_back(num_pixels_fg);
  errors.push_back(num_errors_fg_result);
  errors.push_back(num_pixels_fg_result);
  errors.push_back(num_errors_all);
  errors.push_back(num_pixels_all);
  errors.push_back(num_errors_all_result);
  errors.push_back(num_pixels_all_result);

  // push back density
  errors.push_back((float)num_pixels_all_result/max((float)num_pixels_all,1.0f));

  // return errors
  return errors;
}

vector<float> flowErrorsOutlier (FlowImage &F_gt,FlowImage &F_orig,FlowImage &F_ipol,IntegerImage &O_map) {

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
  int32_t num_errors_bg = 0;
  int32_t num_pixels_bg = 0;
  int32_t num_errors_bg_result = 0;
  int32_t num_pixels_bg_result = 0;
  int32_t num_errors_fg = 0;
  int32_t num_pixels_fg = 0;
  int32_t num_errors_fg_result = 0;
  int32_t num_pixels_fg_result = 0;
  int32_t num_errors_all = 0;
  int32_t num_pixels_all = 0;
  int32_t num_errors_all_result = 0;
  int32_t num_pixels_all_result = 0;

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {
      float fu = F_gt.getFlowU(u,v)-F_ipol.getFlowU(u,v);
      float fv = F_gt.getFlowV(u,v)-F_ipol.getFlowV(u,v);
      float f_dist = sqrt(fu*fu+fv*fv);
      float f_mag  = F_gt.getFlowMagnitude(u,v);
      bool  f_err  = f_dist>ABS_THRESH && f_dist/f_mag>REL_THRESH;
      if (O_map.getValue(u,v)==0) {
        if (F_gt.isValid(u,v)) {
          if (f_err)
            num_errors_bg++;
          num_pixels_bg++;
          if (F_orig.isValid(u,v)) {
            if (f_err)
              num_errors_bg_result++;
            num_pixels_bg_result++;
          }
        }
      } else {
        if (F_gt.isValid(u,v)) {
          if (f_err)
            num_errors_fg++;
          num_pixels_fg++;
          if (F_orig.isValid(u,v)) {
            if (f_err)
              num_errors_fg_result++;
            num_pixels_fg_result++;
          }
        }
      }
      if (F_gt.isValid(u,v)) {
        if (f_err)
          num_errors_all++;
        num_pixels_all++;
        if (F_orig.isValid(u,v)) {
          if (f_err)
            num_errors_all_result++;
          num_pixels_all_result++;
        }
      }
    }
  }

  // push back errors and pixel count
  errors.push_back(num_errors_bg);
  errors.push_back(num_pixels_bg);
  errors.push_back(num_errors_bg_result);
  errors.push_back(num_pixels_bg_result);
  errors.push_back(num_errors_fg);
  errors.push_back(num_pixels_fg);
  errors.push_back(num_errors_fg_result);
  errors.push_back(num_pixels_fg_result);
  errors.push_back(num_errors_all);
  errors.push_back(num_pixels_all);
  errors.push_back(num_errors_all_result);
  errors.push_back(num_pixels_all_result);

  // push back density
  errors.push_back((float)num_pixels_all_result/max((float)num_pixels_all,1.0f));

  // return errors
  return errors;
}

vector<float> sceneFlowErrorsOutlier (DisparityImage &D_gt_0,DisparityImage &D_orig_0,DisparityImage &D_ipol_0,
                                      DisparityImage &D_gt_1,DisparityImage &D_orig_1,DisparityImage &D_ipol_1,
                                      FlowImage &F_gt,FlowImage &F_orig,FlowImage &F_ipol,IntegerImage &O_map) {

  // check file size
  if (D_gt_0.width()!=D_orig_0.width() || D_gt_0.height()!=D_orig_0.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }
  if (D_gt_1.width()!=D_orig_1.width() || D_gt_1.height()!=D_orig_1.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }
  if (F_gt.width()!=F_orig.width() || F_gt.height()!=F_orig.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }

  // extract width and height (all modalities have the same size!)
  int32_t width  = F_gt.width();
  int32_t height = F_gt.height();

  // init errors
  vector<float> errors;
  int32_t num_errors_bg = 0;
  int32_t num_pixels_bg = 0;
  int32_t num_errors_bg_result = 0;
  int32_t num_pixels_bg_result = 0;
  int32_t num_errors_fg = 0;
  int32_t num_pixels_fg = 0;
  int32_t num_errors_fg_result = 0;
  int32_t num_pixels_fg_result = 0;
  int32_t num_errors_all = 0;
  int32_t num_pixels_all = 0;
  int32_t num_errors_all_result = 0;
  int32_t num_pixels_all_result = 0;

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {
      float d_gt_0  = D_gt_0.getDisp(u,v);
      float d_est_0 = D_ipol_0.getDisp(u,v);
      bool  d_err_0 = fabs(d_gt_0-d_est_0)>ABS_THRESH && fabs(d_gt_0-d_est_0)/fabs(d_gt_0)>REL_THRESH;
      float d_gt_1  = D_gt_1.getDisp(u,v);
      float d_est_1 = D_ipol_1.getDisp(u,v);
      bool  d_err_1 = fabs(d_gt_1-d_est_1)>ABS_THRESH && fabs(d_gt_1-d_est_1)/fabs(d_gt_1)>REL_THRESH;
      float fu = F_gt.getFlowU(u,v)-F_ipol.getFlowU(u,v);
      float fv = F_gt.getFlowV(u,v)-F_ipol.getFlowV(u,v);
      float f_dist = sqrt(fu*fu+fv*fv);
      float f_mag  = F_gt.getFlowMagnitude(u,v);
      bool  f_err  = f_dist>ABS_THRESH && f_dist/f_mag>REL_THRESH;
      bool  sf_err = d_err_0 || d_err_1 || f_err;
      if (O_map.getValue(u,v)==0) {
        if (D_gt_0.isValid(u,v) && D_gt_1.isValid(u,v) && F_gt.isValid(u,v)) {
          if (sf_err)
            num_errors_bg++;
          num_pixels_bg++;
          if (D_orig_0.isValid(u,v)) {
            if (sf_err)
              num_errors_bg_result++;
            num_pixels_bg_result++;
          }
        }
      } else {
        if (D_gt_0.isValid(u,v) && D_gt_1.isValid(u,v) && F_gt.isValid(u,v)) {
          if (sf_err)
            num_errors_fg++;
          num_pixels_fg++;
          if (D_orig_0.isValid(u,v)) {
            if (sf_err)
              num_errors_fg_result++;
            num_pixels_fg_result++;
          }
        }
      }
      if (D_gt_0.isValid(u,v) && D_gt_1.isValid(u,v) && F_gt.isValid(u,v)) {
        if (sf_err)
          num_errors_all++;
        num_pixels_all++;
        if (D_orig_0.isValid(u,v)) {
          if (sf_err)
            num_errors_all_result++;
          num_pixels_all_result++;
        }
      }
    }
  }

  // push back errors and pixel count
  errors.push_back(num_errors_bg);
  errors.push_back(num_pixels_bg);
  errors.push_back(num_errors_bg_result);
  errors.push_back(num_pixels_bg_result);
  errors.push_back(num_errors_fg);
  errors.push_back(num_pixels_fg);
  errors.push_back(num_errors_fg_result);
  errors.push_back(num_pixels_fg_result);
  errors.push_back(num_errors_all);
  errors.push_back(num_pixels_all);
  errors.push_back(num_errors_all_result);
  errors.push_back(num_pixels_all_result);

  // push back density
  errors.push_back((float)num_pixels_all_result/max((float)num_pixels_all,1.0f));

  // return errors
  return errors;
}

void writeSceneFlowErrorImage(DisparityImage &D_gt_noc_0,DisparityImage &D_gt_occ_0,DisparityImage &D_ipol_0,
                              DisparityImage &D_gt_noc_1,DisparityImage &D_gt_occ_1,DisparityImage &D_ipol_1,
                              FlowImage &F_gt_noc,FlowImage &F_gt_occ,FlowImage &F_ipol,
                              string file_name) {
  int32_t width = F_gt_noc.width();
  int32_t height = F_gt_noc.height();
  png::image<png::rgb_pixel> image(width,height);
  for (int32_t v=1; v<height-1; v++) {
    for (int32_t u=1; u<width-1; u++) {
      if (F_gt_occ.isValid(u,v)) {
        png::rgb_pixel val;
        float n_err = 0;
        float d_err = fabs(D_ipol_0.getDisp(u,v)-D_gt_occ_0.getDisp(u,v));
        float d_mag = fabs(D_gt_occ_0.getDisp(u,v));
        n_err = std::max(n_err,std::min(d_err/3.0f,20.0f*d_err/d_mag));
        d_err = fabs(D_ipol_1.getDisp(u,v)-D_gt_occ_1.getDisp(u,v));
        d_mag = fabs(D_gt_occ_1.getDisp(u,v));
        n_err = std::max(n_err,std::min(d_err/3.0f,20.0f*d_err/d_mag));
        float dfu = F_ipol.getFlowU(u,v)-F_gt_occ.getFlowU(u,v);
        float dfv = F_ipol.getFlowV(u,v)-F_gt_occ.getFlowV(u,v);
        float f_err = sqrt(dfu*dfu+dfv*dfv);
        float f_mag = F_gt_occ.getFlowMagnitude(u,v);
        n_err = std::max(n_err,std::min(f_err/3.0f,20.0f*f_err/f_mag));
        for (int32_t i=0; i<10; i++) {
          if (n_err>=LC[i][0] && n_err<LC[i][1]) {
            val.red   = (uint8_t)LC[i][2];
            val.green = (uint8_t)LC[i][3];
            val.blue  = (uint8_t)LC[i][4];
          }
        }
        if (!F_gt_noc.isValid(u,v)) {
          val.red   *= 0.5;
          val.green *= 0.5;
          val.blue  *= 0.5;
        }
        for (int32_t v2=v-1; v2<=v+1; v2++)
          for (int32_t u2=u-1; u2<=u+1; u2++)  
            image.set_pixel(u2,v2,val);
      }
    }
  }
  image.write(file_name);
}

bool resultsAvailable (string dir,string folder_name,Mail* mail) {
  int32_t count = 0;
  for (int32_t i=0; i<NUM_TEST_IMAGES; i++) {
    char prefix[256];
    sprintf(prefix,"%06d_10",i);
    FILE *tmp_file = fopen((dir + "/" + prefix + ".png").c_str(),"rb");
    if (tmp_file) {
      count++;
      fclose(tmp_file);
    }
  }
  mail->msg("Found %d/%d images in %s folder.",count,NUM_TEST_IMAGES,folder_name.c_str());
  return count==NUM_TEST_IMAGES;
}

bool eval (string result_sha,Mail* mail) {

  // ground truth and result directories
  string gt_img_dir = "data/scene_flow/image_2";
  string gt_obj_map_dir = "data/scene_flow/obj_map";
  string gt_disp_noc_0_dir = "data/scene_flow/disp_noc_0";
  string gt_disp_occ_0_dir = "data/scene_flow/disp_occ_0";
  string gt_disp_noc_1_dir = "data/scene_flow/disp_noc_1";
  string gt_disp_occ_1_dir = "data/scene_flow/disp_occ_1";
  string gt_flow_noc_dir = "data/scene_flow/flow_noc";
  string gt_flow_occ_dir = "data/scene_flow/flow_occ";
  string result_dir = "results/" + result_sha;
  string result_disp_0_dir = result_dir + "/data/disp_0";
  string result_disp_1_dir = result_dir + "/data/disp_1";
  string result_flow_dir = result_dir + "/data/flow";

  // check availability of results
  bool avail_disp_0 = resultsAvailable(result_disp_0_dir,"disp_0",mail);
  bool avail_disp_1 = resultsAvailable(result_disp_1_dir,"disp_1",mail);
  bool avail_flow   = resultsAvailable(result_flow_dir,"flow",mail);

  // which benchmarks can be evaluated?
  bool eval_disp = avail_disp_0;
  bool eval_flow = avail_flow;
  bool eval_scene_flow = avail_disp_0 && avail_disp_1 && avail_flow;
  if (eval_disp) mail->msg("Evaluating stereo results.");
  if (eval_flow) mail->msg("Evaluating flow results.");
  if (eval_scene_flow) mail->msg("Evaluating scene flow results.");

  // make sure we have something to evaluate at all
  if (!eval_disp && !eval_flow && !eval_scene_flow) {
    mail->msg("Not enough result images found for any of the evaluations, stopping evaluation.");
  	return false;
  }

  // create output directories (depending on which benchmarks to evaluate)
  system(("mkdir " + result_dir + "/image_0/").c_str());
  if (eval_disp) {
    system(("mkdir " + result_dir + "/errors_disp_noc_0/").c_str());
    system(("mkdir " + result_dir + "/errors_disp_occ_0/").c_str());
    system(("mkdir " + result_dir + "/errors_disp_img_0/").c_str());
    system(("mkdir " + result_dir + "/result_disp_img_0/").c_str());
  }
  if (eval_flow) {
    system(("mkdir " + result_dir + "/errors_flow_noc/").c_str());
    system(("mkdir " + result_dir + "/errors_flow_occ/").c_str());
    system(("mkdir " + result_dir + "/errors_flow_img/").c_str());
    system(("mkdir " + result_dir + "/result_flow_img/").c_str());
  }
  if (eval_scene_flow) {
    system(("mkdir " + result_dir + "/errors_disp_noc_1/").c_str());
    system(("mkdir " + result_dir + "/errors_disp_occ_1/").c_str());
    system(("mkdir " + result_dir + "/errors_disp_img_1/").c_str());
    system(("mkdir " + result_dir + "/result_disp_img_1/").c_str());
    system(("mkdir " + result_dir + "/errors_scene_flow_noc/").c_str());
    system(("mkdir " + result_dir + "/errors_scene_flow_occ/").c_str());
    system(("mkdir " + result_dir + "/errors_scene_flow_img/").c_str());
  }

  // accumulators
  float errors_disp_noc_0[3*4]     = {0,0,0,0,0,0,0,0,0,0,0,0};
  float errors_disp_occ_0[3*4]     = {0,0,0,0,0,0,0,0,0,0,0,0};
  float errors_disp_noc_1[3*4]     = {0,0,0,0,0,0,0,0,0,0,0,0};
  float errors_disp_occ_1[3*4]     = {0,0,0,0,0,0,0,0,0,0,0,0};
  float errors_flow_noc[3*4]       = {0,0,0,0,0,0,0,0,0,0,0,0};
  float errors_flow_occ[3*4]       = {0,0,0,0,0,0,0,0,0,0,0,0};
  float errors_scene_flow_noc[3*4] = {0,0,0,0,0,0,0,0,0,0,0,0};
  float errors_scene_flow_occ[3*4] = {0,0,0,0,0,0,0,0,0,0,0,0};

  // for all test files do
  for (int32_t i=0; i<NUM_TEST_IMAGES; i++) {

    // file name
    char prefix[256];
    sprintf(prefix,"%06d_10",i);
    
    // output
    mail->msg("Processing: %s.png",prefix);

    // catch errors, when loading fails
    try {

      // declaration of global data structures
      DisparityImage D_gt_noc_0, D_gt_occ_0, D_orig_0, D_ipol_0;
      DisparityImage D_gt_noc_1, D_gt_occ_1, D_orig_1, D_ipol_1;
      FlowImage F_gt_noc, F_gt_occ, F_orig, F_ipol;

      // load object map (0:background, >0:foreground)
      IntegerImage O_map = IntegerImage(gt_obj_map_dir + "/" + prefix + ".png");

      // copy left camera image 
      if (i<NUM_ERROR_IMAGES) {       
        string img_src = gt_img_dir + "/" + prefix + ".png";
        string img_dst = result_dir + "/image_0/" + prefix + ".png";
        system(("cp " + img_src + " " + img_dst).c_str());
      }

      ///////////////////////////////////////////////////////////////////////////////////////////
      // evaluation of disp 0
      if (eval_disp) {
        
        // load ground truth disparity maps
        D_gt_noc_0 = DisparityImage(gt_disp_noc_0_dir + "/" + prefix + ".png");
        D_gt_occ_0 = DisparityImage(gt_disp_occ_0_dir + "/" + prefix + ".png");

        // check submitted result
        string image_file = result_disp_0_dir + "/" + prefix + ".png";
        if (!imageFormat(image_file,png::color_type_gray,16,D_gt_noc_0.width(),D_gt_noc_0.height())) {
          mail->msg("ERROR: Input must be png, 1 channel, 16 bit, %d x %d px",D_gt_noc_0.width(),D_gt_noc_0.height());
          return false;        
        }

        // load submitted result and interpolate missing values
        D_orig_0 = DisparityImage(image_file);
        D_ipol_0 = DisparityImage(D_orig_0);
        D_ipol_0.interpolateBackground();

        // calculate disparity errors
        vector<float> errors_noc_curr = disparityErrorsOutlier(D_gt_noc_0,D_orig_0,D_ipol_0,O_map);
        vector<float> errors_occ_curr = disparityErrorsOutlier(D_gt_occ_0,D_orig_0,D_ipol_0,O_map);

        // accumulate errors
        for (int32_t j=0; j<errors_noc_curr.size()-1; j++) {
          errors_disp_noc_0[j] += errors_noc_curr[j];
          errors_disp_occ_0[j] += errors_occ_curr[j];
        }

        // save error images
        if (i<NUM_ERROR_IMAGES) {

          // save errors of error images to text file
          FILE *errors_noc_file = fopen((result_dir + "/errors_disp_noc_0/" + prefix + ".txt").c_str(),"w");
          FILE *errors_occ_file = fopen((result_dir + "/errors_disp_occ_0/" + prefix + ".txt").c_str(),"w");
          for (int32_t i=0; i<12; i+=2)
            fprintf(errors_noc_file,"%f ",errors_noc_curr[i]/max(errors_noc_curr[i+1],1.0f));
          fprintf(errors_noc_file,"%f ",errors_noc_curr[12]);
          for (int32_t i=0; i<12; i+=2)
            fprintf(errors_occ_file,"%f ",errors_occ_curr[i]/max(errors_occ_curr[i+1],1.0f));
          fprintf(errors_occ_file,"%f ",errors_occ_curr[12]);
          fclose(errors_noc_file);
          fclose(errors_occ_file);

          // save error image
          D_ipol_0.errorImage(D_gt_noc_0,D_gt_occ_0,true).write(result_dir + "/errors_disp_img_0/" + prefix + ".png");

          // compute maximum disparity
          float max_disp = D_gt_occ_0.maxDisp();

          // save interpolated disparity image false color coded
          D_ipol_0.writeColor(result_dir + "/result_disp_img_0/" + prefix + ".png",max_disp);
        }
      }

      ///////////////////////////////////////////////////////////////////////////////////////////
      // evaluation of flow
      if (eval_flow) {

        // load ground truth flow maps
        F_gt_noc = FlowImage(gt_flow_noc_dir + "/" + prefix + ".png");
        F_gt_occ = FlowImage(gt_flow_occ_dir + "/" + prefix + ".png");

        // check submitted result
        string image_file = result_flow_dir + "/" + prefix + ".png";
        if (!imageFormat(image_file,png::color_type_rgb,16,F_gt_noc.width(),F_gt_noc.height())) {
          mail->msg("ERROR: Input must be png, 3 channels, 16 bit, %d x %d px",F_gt_noc.width(),F_gt_noc.height());
          return false;        
        }

        // load submitted result and interpolate missing values
        F_orig = FlowImage(image_file);
        F_ipol = FlowImage(F_orig); 
        F_ipol.interpolateBackground();     

        // calculate flow errors
        vector<float> errors_noc_curr = flowErrorsOutlier(F_gt_noc,F_orig,F_ipol,O_map);
        vector<float> errors_occ_curr = flowErrorsOutlier(F_gt_occ,F_orig,F_ipol,O_map);
        
        // accumulate errors
        for (int32_t j=0; j<errors_noc_curr.size()-1; j++) {
          errors_flow_noc[j] += errors_noc_curr[j];
          errors_flow_occ[j] += errors_occ_curr[j];
        }

        // save error images
        if (i<NUM_ERROR_IMAGES) {

          // save errors of error images to text file
          FILE *errors_noc_file = fopen((result_dir + "/errors_flow_noc/" + prefix + ".txt").c_str(),"w");
          FILE *errors_occ_file = fopen((result_dir + "/errors_flow_occ/" + prefix + ".txt").c_str(),"w");
          for (int32_t i=0; i<12; i+=2)
            fprintf(errors_noc_file,"%f ",errors_noc_curr[i]/max(errors_noc_curr[i+1],1.0f));
          fprintf(errors_noc_file,"%f ",errors_noc_curr[12]);
          for (int32_t i=0; i<12; i+=2)
            fprintf(errors_occ_file,"%f ",errors_occ_curr[i]/max(errors_occ_curr[i+1],1.0f));
          fprintf(errors_occ_file,"%f ",errors_occ_curr[12]);
          fclose(errors_noc_file);
          fclose(errors_occ_file);

          // save error image
          F_ipol.errorImage(F_gt_noc,F_gt_occ,true).write(result_dir + "/errors_flow_img/" + prefix + ".png");

          // find maximum ground truth flow
          float max_flow = F_gt_occ.maxFlow();

          // save interpolated flow image false color coded
          F_ipol.writeColor(result_dir + "/result_flow_img/" + prefix + ".png",max_flow);
        }
      }

      ///////////////////////////////////////////////////////////////////////////////////////////
      // evaluation of scene flow
      if (eval_scene_flow) {

        // load ground truth disparity maps
        D_gt_noc_1 = DisparityImage(gt_disp_noc_1_dir + "/" + prefix + ".png");
        D_gt_occ_1 = DisparityImage(gt_disp_occ_1_dir + "/" + prefix + ".png");

        // check submitted result
        string image_file = result_disp_1_dir + "/" + prefix + ".png";
        if (!imageFormat(image_file,png::color_type_gray,16,D_gt_noc_1.width(),D_gt_noc_1.height())) {
          mail->msg("ERROR: Input must be png, 1 channel, 16 bit, %d x %d px",D_gt_noc_1.width(),D_gt_noc_1.height());
          return false;        
        }

        // load submitted result and interpolate missing values
        D_orig_1 = DisparityImage(image_file);
        D_ipol_1 = DisparityImage(D_orig_1);
        D_ipol_1.interpolateBackground();

        // calculate disparity errors
        vector<float> errors_noc_curr = disparityErrorsOutlier(D_gt_noc_1,D_orig_1,D_ipol_1,O_map);
        vector<float> errors_occ_curr = disparityErrorsOutlier(D_gt_occ_1,D_orig_1,D_ipol_1,O_map);

        // accumulate errors
        for (int32_t j=0; j<errors_noc_curr.size()-1; j++) {
          errors_disp_noc_1[j] += errors_noc_curr[j];
          errors_disp_occ_1[j] += errors_occ_curr[j];
        }

        // save error images
        if (i<NUM_ERROR_IMAGES) {

          // save errors of error images to text file
          FILE *errors_noc_file = fopen((result_dir + "/errors_disp_noc_1/" + prefix + ".txt").c_str(),"w");
          FILE *errors_occ_file = fopen((result_dir + "/errors_disp_occ_1/" + prefix + ".txt").c_str(),"w");
          for (int32_t i=0; i<12; i+=2)
            fprintf(errors_noc_file,"%f ",errors_noc_curr[i]/max(errors_noc_curr[i+1],1.0f));
          fprintf(errors_noc_file,"%f ",errors_noc_curr[12]);
          for (int32_t i=0; i<12; i+=2)
            fprintf(errors_occ_file,"%f ",errors_occ_curr[i]/max(errors_occ_curr[i+1],1.0f));
          fprintf(errors_occ_file,"%f ",errors_occ_curr[12]);
          fclose(errors_noc_file);
          fclose(errors_occ_file);

          // save error image
          D_ipol_1.errorImage(D_gt_noc_1,D_gt_occ_1,true).write(result_dir + "/errors_disp_img_1/" + prefix + ".png");

          // compute maximum disparity
          float max_disp = D_gt_occ_1.maxDisp();

          // save interpolated disparity image false color coded
          D_ipol_1.writeColor(result_dir + "/result_disp_img_1/" + prefix + ".png",max_disp);
        }

        // calculate scene flow errors
        errors_noc_curr = sceneFlowErrorsOutlier(D_gt_noc_0,D_orig_0,D_ipol_0,D_gt_noc_1,D_orig_1,D_ipol_1,F_gt_noc,F_orig,F_ipol,O_map);
        errors_occ_curr = sceneFlowErrorsOutlier(D_gt_occ_0,D_orig_0,D_ipol_0,D_gt_occ_1,D_orig_1,D_ipol_1,F_gt_occ,F_orig,F_ipol,O_map);

        // accumulate errors
        for (int32_t j=0; j<errors_noc_curr.size()-1; j++) {
          errors_scene_flow_noc[j] += errors_noc_curr[j];
          errors_scene_flow_occ[j] += errors_occ_curr[j];
        }

        // save error images
        if (i<NUM_ERROR_IMAGES) {

          // save errors of error images to text file
          FILE *errors_noc_file = fopen((result_dir + "/errors_scene_flow_noc/" + prefix + ".txt").c_str(),"w");
          FILE *errors_occ_file = fopen((result_dir + "/errors_scene_flow_occ/" + prefix + ".txt").c_str(),"w");
          for (int32_t i=0; i<12; i+=2)
            fprintf(errors_noc_file,"%f ",errors_noc_curr[i]/max(errors_noc_curr[i+1],1.0f));
          fprintf(errors_noc_file,"%f ",errors_noc_curr[12]);
          for (int32_t i=0; i<12; i+=2)
            fprintf(errors_occ_file,"%f ",errors_occ_curr[i]/max(errors_occ_curr[i+1],1.0f));
          fprintf(errors_occ_file,"%f ",errors_occ_curr[12]);
          fclose(errors_noc_file);
          fclose(errors_occ_file);

          // save error image
          writeSceneFlowErrorImage(D_gt_noc_0,D_gt_occ_0,D_ipol_0,D_gt_noc_1,D_gt_occ_1,D_ipol_1,
                                   F_gt_noc,F_gt_occ,F_ipol,result_dir + "/errors_scene_flow_img/" + prefix + ".png");
        }
      }

    // on error, exit
    } catch (...) {
      mail->msg("ERROR: Couldn't read: %s.png",prefix);
      return false;
    }
  }

  string stats_file_name;
  FILE *stats_file;

  // write summary statistics for disparity evaluation
  if (eval_disp) {
    stats_file_name = result_dir + "/stats_disp_noc_0.txt";
    stats_file = fopen(stats_file_name.c_str(),"w");
    for (int32_t i=0; i<12; i+=2)
      fprintf(stats_file,"%f ",errors_disp_noc_0[i]/max(errors_disp_noc_0[i+1],1.0f));
    fprintf(stats_file,"%f ",errors_disp_noc_0[11]/max(errors_disp_noc_0[9],1.0f));
    fprintf(stats_file,"\n");
    fclose(stats_file);
    stats_file_name = result_dir + "/stats_disp_occ_0.txt";
    stats_file = fopen(stats_file_name.c_str(),"w");
    for (int32_t i=0; i<12; i+=2)
      fprintf(stats_file,"%f ",errors_disp_occ_0[i]/max(errors_disp_occ_0[i+1],1.0f));
    fprintf(stats_file,"%f ",errors_disp_occ_0[11]/max(errors_disp_occ_0[9],1.0f));
    fprintf(stats_file,"\n");
    fclose(stats_file);
  }

  // write summary statistics for flow evaluation
  if (eval_flow) {
    stats_file_name = result_dir + "/stats_flow_noc.txt";
    stats_file = fopen(stats_file_name.c_str(),"w");
    for (int32_t i=0; i<12; i+=2)
      fprintf(stats_file,"%f ",errors_flow_noc[i]/max(errors_flow_noc[i+1],1.0f));
    fprintf(stats_file,"%f ",errors_flow_noc[11]/max(errors_flow_noc[9],1.0f));
    fprintf(stats_file,"\n");
    fclose(stats_file);
    stats_file_name = result_dir + "/stats_flow_occ.txt";
    stats_file = fopen(stats_file_name.c_str(),"w");
    for (int32_t i=0; i<12; i+=2)
      fprintf(stats_file,"%f ",errors_flow_occ[i]/max(errors_flow_occ[i+1],1.0f)); 
    fprintf(stats_file,"%f ",errors_flow_occ[11]/max(errors_flow_occ[9],1.0f));
    fprintf(stats_file,"\n");
    fclose(stats_file);
  }

  // write summary statistics for scene flow evaluation
  if (eval_scene_flow) {
    stats_file_name = result_dir + "/stats_disp_noc_1.txt";
    stats_file = fopen(stats_file_name.c_str(),"w");
    for (int32_t i=0; i<12; i+=2)
      fprintf(stats_file,"%f ",errors_disp_noc_1[i]/max(errors_disp_noc_1[i+1],1.0f));
    fprintf(stats_file,"%f ",errors_disp_noc_1[11]/max(errors_disp_noc_1[9],1.0f));
    fprintf(stats_file,"\n");
    fclose(stats_file);
    stats_file_name = result_dir + "/stats_disp_occ_1.txt";
    stats_file = fopen(stats_file_name.c_str(),"w");
    for (int32_t i=0; i<12; i+=2)
      fprintf(stats_file,"%f ",errors_disp_occ_1[i]/max(errors_disp_occ_1[i+1],1.0f)); 
    fprintf(stats_file,"%f ",errors_disp_occ_1[11]/max(errors_disp_occ_1[9],1.0f)); 
    fprintf(stats_file,"\n");
    fclose(stats_file);
    stats_file_name = result_dir + "/stats_scene_flow_noc.txt";
    stats_file = fopen(stats_file_name.c_str(),"w");
    for (int32_t i=0; i<12; i+=2)
      fprintf(stats_file,"%f ",errors_scene_flow_noc[i]/max(errors_scene_flow_noc[i+1],1.0f));
    fprintf(stats_file,"%f ",errors_scene_flow_noc[11]/max(errors_scene_flow_noc[9],1.0f));
    fprintf(stats_file,"\n");
    stats_file_name = result_dir + "/stats_scene_flow_occ.txt";
    stats_file = fopen(stats_file_name.c_str(),"w");
    for (int32_t i=0; i<12; i+=2)
      fprintf(stats_file,"%f ",errors_scene_flow_occ[i]/max(errors_scene_flow_occ[i+1],1.0f)); 
    fprintf(stats_file,"%f ",errors_scene_flow_occ[11]/max(errors_scene_flow_occ[9],1.0f)); 
    fprintf(stats_file,"\n");
    fclose(stats_file);
  }

  // success
	return true;
}

int32_t main (int32_t argc,char *argv[]) {

  // we need 2 or 4 arguments!
  if (argc!=2 && argc!=4) {
    cout << "Usage: ./eval_scene_flow result_sha [user_sha email]" << endl;
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
  if (argc==4) mail->finalize(success,"scene_flow",result_sha,argv[2]);
  else         mail->finalize(success,"scene_flow",result_sha);

  // send mail and exit
  delete mail;
  return 0;
}


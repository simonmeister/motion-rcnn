#ifndef MAIL_H
#define MAIL_H

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

class Mail {

public:

  Mail (std::string email = "",std::string from = "noreply@cvlibs.net",std::string subject = "KITTI Evaluation Benchmark") {
    if (email.compare("")) {
      char cmd[2048];
      sprintf(cmd,"/usr/lib/sendmail -t -f noreply@cvlibs.net");
      mail = popen(cmd,"w");
      fprintf(mail,"To: %s\n", email.c_str());
      fprintf(mail,"From: %s\n", from.c_str());
      fprintf(mail,"Subject: %s\n", subject.c_str());
      fprintf(mail,"\n\n");
    } else {
      mail = 0;
    }
  }
  
  ~Mail() {
    if (mail) {
      pclose(mail);
    }
  }
  
  void msg (const char *format, ...) {
    va_list args;
    va_start(args,format);
    if (mail) {
      vfprintf(mail,format,args);
      fprintf(mail,"\n");
    }
    vprintf(format,args);
    printf("\n");
    va_end(args);
  }
  
  void msg (std::string str) {
    if (mail) {
      fprintf(mail,"%s\n",str.c_str());
    }
    printf("%s\n",str.c_str());
  }

  void finalize (bool success,std::string benchmark,std::string result_sha="",std::string user_sha="") {
    if (success) {
      msg("Your evaluation results are available at:");
      msg("http://www.cvlibs.net/datasets/kitti/user_submit_check_login.php?benchmark=%s&user=%s&result=%s",benchmark.c_str(),user_sha.c_str(), result_sha.c_str());
    } else {
      msg("An error occured while processing your results.");
      msg("Please make sure that the data in your zip archive has the right format!");
    }
  }
    
private:

  FILE *mail;
  
};

#endif

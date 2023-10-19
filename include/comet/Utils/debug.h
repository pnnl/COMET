#undef comet_debug
#undef comet_pdump
#undef comet_vdump

#ifdef COMET_DEBUG_MODE
#define comet_debug() llvm::errs() << __FILE__ << ":" << __LINE__ << " "
#define comet_pdump(n)                                \
  llvm::errs() << __FILE__ << ":" << __LINE__ << " "; \
  n->dump()
#define comet_vdump(n)                                \
  llvm::errs() << __FILE__ << ":" << __LINE__ << " "; \
  n.dump()
#else
#define comet_debug() if(true){}else llvm::errs()
#define comet_pdump(n)
#define comet_vdump(n)
#endif
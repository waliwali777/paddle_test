/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/profiler/host_tracer.h"
#include "glog/logging.h"
#include "paddle/fluid/platform/profiler/host_event_recorder.h"

namespace paddle {
namespace platform {

namespace {

void ProcessHostEvents(const HostEventSection& host_events,
                       TraceEventCollector* collector) {
  for (const auto& thr_sec : host_events.thr_sections) {
    uint64_t tid = thr_sec.thread_id;
    for (const auto& evt : thr_sec.events) {
      HostEvent event;
      event.name = evt.name;
      event.start_ns = evt.start_ns;
      event.end_ns = evt.end_ns;
      event.thread_id = tid;
      collector->AddHostEvent(std::move(event));
    }
  }
}
}  // namespace

void HostTracer::StartTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::READY || state_ == TracerState::STOPED, true,
      platform::errors::PreconditionNotMet("TracerState must be READY"));
  auto& recorder = HostEventRecorder::GetInstance();
  recorder.GatherEvents();
  recorder.StartTrace(trace_level_);
  state_ = TracerState::STARTED;
}

void HostTracer::StopTracing() {
  PADDLE_ENFORCE_EQ(
      state_, TracerState::STARTED,
      platform::errors::PreconditionNotMet("TracerState must be STARTED"));
  HostEventRecorder::GetInstance().StopTrace();
  state_ = TracerState::STOPED;
}

void HostTracer::CollectTraceData(TraceEventCollector* collector) {
  PADDLE_ENFORCE_EQ(
      state_, TracerState::STOPED,
      platform::errors::PreconditionNotMet("TracerState must be STOPED"));
  HostEventSection host_events =
      HostEventRecorder::GetInstance().GatherEvents();
  ProcessHostEvents(host_events, collector);
}

}  // namespace platform
}  // namespace paddle

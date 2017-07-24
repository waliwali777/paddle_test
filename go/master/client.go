// Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package master

import (
	"os"
	"time"

	"github.com/PaddlePaddle/Paddle/go/connection"
	"github.com/PaddlePaddle/recordio"
	log "github.com/sirupsen/logrus"
)

// Client is the client of the master server.
type Client struct {
	conn *connection.Conn
	ch   chan record
}

type record struct {
	r   []byte
	err error
}

// NewClient creates a new Client.
//
// bufSize is the record buffer size. NextRecord will read from this
// buffer.
func NewClient(addrCh <-chan string, bufSize int) *Client {
	c := &Client{}
	c.conn = connection.New()
	c.ch = make(chan record, bufSize)
	go c.monitorMaster(addrCh)
	// FIXME: async connection creation
	time.Sleep(time.Second)
	err := c.addClient()
	if err != nil {
		log.Errorln("init client(addClient) error:", err)
	}
	return c
}

func (c *Client) getRecords() {
	for {
		t, err := c.getTask()
		if err != nil {
			if err.Error() == ErrAllTaskFinishError.Error() || err.Error() == ErrNoMoreAvailableError.Error() {
				log.Infof("Got %v, stopping getRecords routine.", err)
				c.ch <- record{nil, err}
				return
			}
			log.Errorf("getTask error: %s", err)
		}

		for _, chunk := range t.Chunks {
			f, e := os.Open(chunk.Path)
			if e != nil {
				log.Errorln(e)
				continue
			}

			s := recordio.NewRangeScanner(f, &chunk.Index, -1, -1)
			for s.Scan() {
				c.ch <- record{s.Record(), nil}
			}

			if s.Err() != nil {
				c.ch <- record{nil, s.Err()}
				log.Errorln(err, chunk.Path)
			}

			err = f.Close()
			if err != nil {
				log.Errorln(err)
			}
		}

		// We treat a task as finished whenever the last data
		// instance of the task is read. This is not exactly
		// correct, but a reasonable approximation.
		err = c.taskFinished(t.Meta.ID)
		if err != nil {
			log.Errorln(err)
		}
	}
}

func (c *Client) monitorMaster(addrCh <-chan string) {
	lastMaster := ""
	for curMaster := range addrCh {
		// connect to the new address once address changed.
		if curMaster != lastMaster {
			if curMaster == "" {
				err := c.conn.Close()
				if err != nil {
					log.Errorln(err)
				}
			} else {
				err := c.conn.Connect(curMaster)
				if err != nil {
					log.Errorln(err)

					// connect to addr failed, set
					// to last known addr in order
					// to retry next time.
					curMaster = lastMaster
				}
			}
		}
		lastMaster = curMaster
	}
}

// SetDataset sets dataset to dispatch for the master server.
//
// SetDataset can be call multiple times at one pass. But only the first call
// will be honored.
//
// After all tasks are done, another call of SetDataset will start another pass.
func (c *Client) SetDataset(globPaths []string) error {
	err := c.conn.Call("Service.SetDataset", globPaths, nil)
	// start to getRecords go-routine before each pass
	go c.getRecords()
	return err
}

// getTask gets a new task from the master server.
func (c *Client) getTask() (Task, error) {
	var t Task
	err := c.conn.Call("Service.GetTask", 0, &t)
	return t, err
}

// TaskFinished tells the master server a task is finished.
func (c *Client) taskFinished(taskID int) error {
	return c.conn.Call("Service.TaskFinished", taskID, nil)
}

// TaskFailed tell the master server as task is failed.
func (c *Client) taskFailed(meta TaskMeta) error {
	return c.conn.Call("Service.TaskFailed", meta, nil)
}

// NextRecord returns next record in the dataset.
//
// NextRecord will block until the next record is available. It is
// thread-safe.
func (c *Client) NextRecord() ([]byte, error) {
	r := <-c.ch
	if r.err != nil && (r.err.Error() == ErrAllTaskFinishError.Error() || r.err.Error() == ErrNoMoreAvailableError.Error()) {
		err := c.PassFinish()
		if err != nil {
			return nil, err
		}
	}
	return r.r, r.err
}

func (c *Client) addClient() error {
	return c.conn.Call("Service.AddClient", 0, nil)
}

// PassFinish set current pass to finish
func (c *Client) PassFinish() error {
	err := c.conn.Call("Service.PassFinish", 0, nil)
	return err
}

// PassStart reset pass counter.
func (c *Client) PassStart() error {
	return c.conn.Call("Service.PassStart", 0, nil)
}
